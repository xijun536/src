#!/usr/bin/env bash
set -e
workspace_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /home/nuc/miniconda3/etc/profile.d/conda.sh
conda activate d435i
if [[ ! -f "$workspace_dir/install/local_setup.bash" ]]; then
    echo "Workspace not built: $workspace_dir/install/local_setup.bash" >&2
    exit 1
fi
source "$workspace_dir/install/local_setup.bash"
exec ros2 launch d435i_vision d435i.launch.py "$@"
