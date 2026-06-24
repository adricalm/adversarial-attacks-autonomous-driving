#!/usr/bin/env bash
set -u

LOG="$HOME/summer26/logs/01_gui_preflight_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee "$LOG") 2>&1

section() {
  echo
  echo "===== $* ====="
}

section "DATE / HOST / USER"
date
hostname
whoami
id
groups

section "OS / GPU"
lsb_release -a 2>/dev/null || cat /etc/os-release
uname -a
nvidia-smi || true
ls -l /dev/nvidia* 2>/dev/null || true
ls -l /dev/dri 2>/dev/null || true

section "SESSION / DISPLAY ENVIRONMENT"
env | sort | grep -E '^(DISPLAY|WAYLAND_DISPLAY|XDG_|DESKTOP_SESSION|GNOME|KDE|DBUS_SESSION_BUS_ADDRESS|XAUTHORITY|SSH_|QT_|SDL_|VK_)=' || true
echo "uid=$(id -u)"
ls -ld "/run/user/$(id -u)" 2>/dev/null || true
ls -ld /tmp/.X11-unix 2>/dev/null || true
ls -l /tmp/.X11-unix 2>/dev/null || true

section "LOGIND SESSIONS"
loginctl list-sessions 2>/dev/null || true
loginctl user-status "$USER" 2>/dev/null | head -120 || true

section "GUI / REMOTE DESKTOP PROCESSES"
ps -eo user,pid,ppid,stat,comm,args --sort=user \
  | grep -Ei 'gnome|kde|plasmashell|xfce|xorg|xwayland|wayland|gdm|lightdm|sddm|xrdp|vnc|tigervnc|x11vnc|sunshine|moonlight|pipewire|pulseaudio|dbus' \
  | grep -v grep || true

section "LISTENING GUI-RELATED PORTS"
ss -ltnp 2>/dev/null | grep -E '(:3389|:5900|:5901|:5902|:6080|:47984|:47989|:48010)' || true

section "DISPLAY / GL / VULKAN TESTS"
command -v xdpyinfo >/dev/null && xdpyinfo | head -40 || echo "xdpyinfo unavailable or no DISPLAY"
command -v xrandr >/dev/null && xrandr --listmonitors || echo "xrandr unavailable or no DISPLAY"
command -v glxinfo >/dev/null && glxinfo -B || echo "glxinfo unavailable or no DISPLAY"
command -v vulkaninfo >/dev/null && timeout 10 vulkaninfo --summary || echo "vulkaninfo unavailable or failed"

section "AWSIM FILES"
find "$HOME/summer26" -maxdepth 6 -type f \( -name '*.x86_64' -o -iname '*awsim*' \) -printf '%M %u:%g %s %p\n' 2>/dev/null | sort || true
find "$HOME/summer26/data/awsim" -maxdepth 4 -type f -printf '%M %u:%g %s %p\n' 2>/dev/null | head -200 || true

section "DOCKER READ-ONLY CHECK"
if command -v docker >/dev/null; then
  docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || echo "docker ps without sudo failed"
  sudo -n docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || echo "sudo -n docker ps unavailable/no cached sudo"
fi

section "DONE"
echo "Log saved to $LOG"
