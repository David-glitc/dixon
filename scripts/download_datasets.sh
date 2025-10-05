#!/usr/bin/env bash
set -euo pipefail

# Downloads datasets used by the repo.
# - MNIST gzip files -> repo root
# - Iris CSV -> ml_examples/src/iris.csv

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/.." && pwd)"
data_dir="$repo_root/data"

mnist_base="https://storage.googleapis.com/cvdf-datasets/mnist"
mnist_files=(
  train-images-idx3-ubyte.gz
  train-labels-idx1-ubyte.gz
)

iris_url_default="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris_target="$repo_root/ml_examples/src/iris.csv"

echo "Repo root: $repo_root"

download() {
  local url="$1"; shift
  local out="$1"; shift
  if [[ -f "$out" ]]; then
    echo "Already exists: $out"
    return 0
  fi
  echo "Downloading $url -> $out"
  mkdir -p "$(dirname "$out")"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --connect-timeout 10 -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    echo "Error: curl or wget required." >&2
    exit 1
  fi
}

echo "=== MNIST ==="
for f in "${mnist_files[@]}"; do
  download "$mnist_base/$f" "$data_dir/$f"
done

echo "=== Iris ==="
download "$iris_url_default" "$iris_target"

echo "Done."


