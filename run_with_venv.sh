#!/bin/bash
# Wrapper script to properly activate virtual environment and run Python commands
# Specifically designed to work with Cursor's terminal

# Set the project root
PROJECT_ROOT="/home/balaplumpat/Projects/pmu-instance-sampah-fuzzy"
VENV_PATH="$PROJECT_ROOT/.venv"
SITE_PACKAGES="$VENV_PATH/lib/python3.11/site-packages"

# Force use of virtual environment Python
export VIRTUAL_ENV="$VENV_PATH"
export PATH="$VENV_PATH/bin:$PATH"

# Export PYTHONPATH to include virtual environment packages
export PYTHONPATH="$SITE_PACKAGES:$PYTHONPATH"

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Debug information (uncomment for troubleshooting)
# echo "Using Python: $(which python)"
# echo "PYTHONPATH: $PYTHONPATH"
# echo "VIRTUAL_ENV: $VIRTUAL_ENV"

# Run the command passed as arguments
exec "$@" 