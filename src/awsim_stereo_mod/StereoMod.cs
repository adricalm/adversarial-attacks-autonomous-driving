// StereoMod - adds a second (right) camera to AWSIM Labs v1.6.1 at runtime.
//
// WHY THIS EXISTS
// We only have the prebuilt AWSIM binary, not the Unity project, but Arka's KITTI
// recorder needs a stereo pair. AWSIM already supports multiple cameras natively
// (CameraSensorHolder holds a List<CameraSensor>), and the scene simply ships one. So
// instead of rebuilding AWSIM in Unity we clone the existing camera at runtime.
//
// Cloning (rather than hand-placing a new camera) is what makes the pair trustworthy:
// the copy inherits rotation, FOV, resolution and clipping planes exactly, so the two
// cameras are parallel and rectified by construction.
//
// HOW IT IS LOADED
// No BepInEx / Doorstop. This assembly is registered in Unity's own
// awsim_labs_Data/RuntimeInitializeOnLoads.json and listed in ScriptingAssemblies.json,
// so Unity calls Bootstrap.Init() itself at startup. See scripts/awsim_stereo_install.py.
//
// GEOMETRY - must match dsgn/datasets/arka/.../calib (all frames identical):
//   fx=960.0  fy=959.3908081054688  cx=960.5  cy=540.5  at 1920x1080
//   P2 Tx = +259.2 -> left  camera centre x = -Tx/fx = -0.27 m
//   P3 Tx = -259.2 -> right camera centre x = +0.27 m   => baseline 0.54 m
// P0 has Tx=0, i.e. the reference frame is the *stock* camera position. So the rig is
// centred on the stock camera: the original moves to -0.27 and the clone to +0.27.
//
// NOTE: this shifts the traffic-light camera 27 cm left of where Autoware's
// awsim_labs_sensor_kit URDF believes it is. Harmless for recording (the recorder uses
// a hardcoded base_link->camera transform, not TF) and for traffic lights that are
// force-set green, but do not rely on that camera for precise projection.
//
// Avoid System.Linq: this build has Unity managed-code stripping enabled and the game's
// System.Core.dll is missing LINQ members (this is what stopped BepInEx from loading).

using System.Collections.Generic;
using System.Reflection;
using AWSIM;
using UnityEngine;

namespace StereoMod
{
    public static class Bootstrap
    {
        // Invoked by Unity via RuntimeInitializeOnLoads.json (loadTypes 0 = AfterSceneLoad).
        public static void Init()
        {
            var go = new GameObject("StereoModRunner");
            Object.DontDestroyOnLoad(go);
            go.AddComponent<StereoRig>();
            Debug.Log("[StereoMod] bootstrap ok (v1)");
        }
    }

    public class StereoRig : MonoBehaviour
    {
        const float HalfBaselineMeters = 0.27f;
        const string RightImageTopic = "/sensing/camera_right/traffic_light/image_raw";
        const string RightInfoTopic = "/sensing/camera_right/traffic_light/camera_info";
        const string RightFrameId = "traffic_light_right_camera/camera_optical_link";

        const BindingFlags Priv = BindingFlags.Instance | BindingFlags.NonPublic;

        readonly List<int> handled = new List<int>();
        float timer;

        // The ego vehicle is spawned only after the map scene finishes loading, so poll
        // rather than assume the holder exists. Also covers AWSIM's F12 scene reload.
        void Update()
        {
            timer += Time.unscaledDeltaTime;
            if (timer < 1.0f) return;
            timer = 0f;

            var holders = Object.FindObjectsOfType<CameraSensorHolder>();
            for (int i = 0; i < holders.Length; i++)
            {
                TryInstall(holders[i]);
            }
        }

        void TryInstall(CameraSensorHolder holder)
        {
            int id = holder.GetInstanceID();
            if (handled.Contains(id)) return;

            var listField = typeof(CameraSensorHolder).GetField("cameraSensors", Priv);
            var queueField = typeof(CameraSensorHolder).GetField("renderInQueue", Priv);
            var camField = typeof(CameraSensor).GetField("cameraObject", Priv);
            if (listField == null || queueField == null || camField == null)
            {
                Debug.LogError("[StereoMod] AWSIM internals differ from v1.6.1; not installing.");
                handled.Add(id);
                return;
            }

            var sensors = listField.GetValue(holder) as List<CameraSensor>;
            if (sensors == null || sensors.Count == 0) return;   // ego not ready yet
            if (sensors.Count > 1)                                // already stereo
            {
                handled.Add(id);
                return;
            }

            var left = sensors[0];
            var leftGo = left.gameObject;
            var cam = camField.GetValue(left) as Camera;
            if (cam == null) return;

            // Render every camera on the SAME frame. With renderInQueue = true the
            // sensors render on consecutive frames and ego motion corrupts disparity.
            queueField.SetValue(holder, false);

            var parent = leftGo.transform.parent;
            var basePos = leftGo.transform.localPosition;

            // Clone under an inactive parent so the clone's Awake()/Start() are deferred
            // until after its ROS topics are changed. CameraRos2Publisher.Awake() creates
            // the publishers from those fields, so ordering matters. This also avoids
            // toggling the original (UICameraBridge has OnEnable/OnDisable side effects).
            var incubator = new GameObject("StereoModIncubator");
            incubator.SetActive(false);

            var rightGo = Object.Instantiate(leftGo, incubator.transform);
            rightGo.name = leftGo.name + "_Right";

            var pub = rightGo.GetComponent<CameraRos2Publisher>();
            if (pub == null)
            {
                Debug.LogError("[StereoMod] clone has no CameraRos2Publisher; aborting.");
                Object.Destroy(rightGo);
                Object.Destroy(incubator);
                handled.Add(id);
                return;
            }
            pub.imageTopic = RightImageTopic;
            pub.cameraInfoTopic = RightInfoTopic;
            pub.frameId = RightFrameId;

            // Only the original should drive the on-screen preview. Disabling before the
            // object goes active means OnEnable never fires. Matched by type name so this
            // does not need a compile-time reference to the UI assembly internals.
            var comps = rightGo.GetComponents<MonoBehaviour>();
            for (int i = 0; i < comps.Length; i++)
            {
                if (comps[i] != null && comps[i].GetType().Name == "UICameraBridge")
                {
                    comps[i].enabled = false;
                }
            }

            // Offset along the *camera's* own right axis, expressed in the parent frame.
            // Doing it this way is correct whatever axis convention the parent uses.
            Vector3 worldRight = cam.transform.right;
            Vector3 offset = worldRight * HalfBaselineMeters;
            Vector3 localOffset = (parent != null)
                ? parent.InverseTransformVector(offset)
                : offset;

            rightGo.transform.localPosition = basePos + localOffset;
            rightGo.transform.localRotation = leftGo.transform.localRotation;
            rightGo.transform.localScale = leftGo.transform.localScale;

            // Reparenting to the (active) real parent is what activates the clone and
            // finally runs Awake()/Start(). worldPositionStays:false keeps the local
            // transform we just set.
            rightGo.transform.SetParent(parent, false);

            leftGo.transform.localPosition = basePos - localOffset;

            var rightSensor = rightGo.GetComponent<CameraSensor>();
            if (rightSensor == null)
            {
                Debug.LogError("[StereoMod] clone has no CameraSensor; aborting.");
                Object.Destroy(rightGo);
                Object.Destroy(incubator);
                handled.Add(id);
                return;
            }

            // CameraSensorHolder re-reads Count every cycle, so appending takes effect live.
            sensors.Add(rightSensor);

            Object.Destroy(incubator);
            handled.Add(id);

            Debug.Log("[StereoMod] installed stereo rig on '" + holder.gameObject.name +
                      "': baseline " + (2f * HalfBaselineMeters).ToString("0.000") + " m, " +
                      "left=" + pub.imageTopic.Replace("camera_right", "camera") + " (stock), " +
                      "right=" + RightImageTopic);
        }
    }
}
