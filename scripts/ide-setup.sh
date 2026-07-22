# Created by TG on 2025-06-28

VSC_EXTS=(
    visualstudioexptteam.vscodeintellicode
    github.copilot
    foxundermoon.shell-format
    ms-python.python
    ms-python.black-formatter
    ms-python.debugpy
    ms-python.vscode-python-envs
)


setup-vscode() {
    set -euo pipefail
    echo "Setting up Visual Studio Code..."

    # Check if Visual Studio Code is installed
    if ! command -v code &>/dev/null; then
        # check if we have vscode server  @  ~/.vscode-server/cli/servers/*/server/bin/remote-cli/code and add it to PATH
        code_bin=$(realpath ~/.vscode-server/cli/servers/*/server/bin/remote-cli/code)
        if [[ -f "$code_bin" ]]; then
            export PATH="$PATH:$(dirname ${code_bin})"
            echo "Added Visual Studio Code server to PATH: $(dirname ${code_bin})"
        fi
    fi
    if command -v code &>/dev/null; then
        echo "Visual Studio Code is available in PATH."
    else
        echo "Visual Studio Code is not installed. Please install and/or add 'code' to PATH first."
        exit 1
    fi

    for ext in "${VSC_EXTS[@]}"; do
        if ! code --list-extensions | grep -q "$ext"; then
            echo "Installing $ext"
            code --install-extension "$ext"
        else
            echo "Already installed: $ext"
        fi
    done
    echo "VS Code setup complete."
}


setup-vscode
