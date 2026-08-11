from pathlib import Path
import argparse
import hashlib
import os
import re
import subprocess
import tarfile

def clean_value(v: str) -> str:
    v = v.strip()
    v = v.strip('"').strip("'")
    return v

def replace_vars(v: str, data_dir: Path) -> Path:
    v = v.replace("{{ data_dir }}", str(data_dir))
    v = v.replace("{{data_dir}}", str(data_dir))
    return Path(os.path.expanduser(v))

def parse_module_blocks(task_file: Path, module: str):
    lines = task_file.read_text().splitlines()
    marker = f"ansible.builtin.{module}:"
    for i, line in enumerate(lines):
        if line.strip() == marker:
            body = []
            j = i + 1
            while j < len(lines) and not lines[j].startswith("- name:"):
                body.append(lines[j])
                j += 1
            yield i + 1, body

def parse_kv(body):
    out = {}
    for line in body:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if ":" in s:
            k, v = s.split(":", 1)
            out[k.strip()] = clean_value(v)
    return out

def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_sha256(path: Path, expected: str):
    actual = sha256sum(path)
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch for {path}\nexpected: {expected}\nactual:   {actual}")

def download(url: str, dest: Path, checksum: str | None):
    dest.parent.mkdir(parents=True, exist_ok=True)

    expected_sha = None
    if checksum and checksum.startswith("sha256:"):
        expected_sha = checksum.split(":", 1)[1]

    if dest.exists() and expected_sha:
        try:
            verify_sha256(dest, expected_sha)
            print(f"SKIP valid: {dest}")
            return
        except Exception:
            print(f"Existing file failed checksum, re-downloading: {dest}")
            dest.unlink()

    elif dest.exists():
        print(f"SKIP exists: {dest}")
        return

    tmp = dest.with_name(dest.name + ".download")
    if tmp.exists():
        tmp.unlink()

    print(f"\nDOWNLOAD: {url}")
    print(f"TO:       {dest}")

    subprocess.run(
        ["curl", "-L", "--fail", "--retry", "3", "--connect-timeout", "30",
         "--progress-bar", url, "-o", str(tmp)],
        check=True,
    )

    if expected_sha:
        verify_sha256(tmp, expected_sha)

    tmp.replace(dest)
    print(f"OK: {dest}")

def safe_extract_tar(tar: tarfile.TarFile, dest: Path):
    dest = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest)):
            raise RuntimeError(f"Unsafe tar path: {member.name}")
    tar.extractall(dest)

def extract_archive(src: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\nEXTRACT: {src}")
    print(f"TO:      {dest}")

    if src.name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(src, "r:*") as tar:
            safe_extract_tar(tar, dest)
    else:
        raise RuntimeError(f"Unsupported archive type: {src}")

    print(f"OK extracted: {src}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    task_file = Path(args.task_file)
    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    get_urls = []
    for line_no, body in parse_module_blocks(task_file, "get_url"):
        kv = parse_kv(body)
        if "url" not in kv or "dest" not in kv:
            raise RuntimeError(f"Could not parse get_url block near line {line_no}")
        get_urls.append({
            "url": kv["url"],
            "dest": replace_vars(kv["dest"], data_dir),
            "checksum": kv.get("checksum"),
        })

    archives = []
    for line_no, body in parse_module_blocks(task_file, "unarchive"):
        kv = parse_kv(body)
        if "src" not in kv or "dest" not in kv:
            raise RuntimeError(f"Could not parse unarchive block near line {line_no}")
        archives.append({
            "src": replace_vars(kv["src"], data_dir),
            "dest": replace_vars(kv["dest"], data_dir),
        })

    print(f"data_dir: {data_dir}")
    print(f"downloads: {len(get_urls)}")
    print(f"archives:  {len(archives)}")

    for item in get_urls:
        download(item["url"], item["dest"], item["checksum"])

    for item in archives:
        extract_archive(item["src"], item["dest"])

    print("\nDONE: all artifacts downloaded and extracted.")

if __name__ == "__main__":
    main()
