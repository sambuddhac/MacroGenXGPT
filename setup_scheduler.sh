#!/bin/bash
# Setup script for Vector Index Auto-Update Scheduler

set -e

echo "Vector Index Scheduler Setup"
echo "============================"
echo ""

# Get the current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SERVICE_FILE="$SCRIPT_DIR/vector-index-scheduler.service"
SYSTEMD_DIR="$HOME/.config/systemd/user"

# Function to show usage
show_usage() {
    echo "Usage: $0 [install|uninstall|status|start|stop|restart|logs]"
    echo ""
    echo "Commands:"
    echo "  install    - Install the scheduler as a systemd user service"
    echo "  uninstall  - Uninstall the scheduler service"
    echo "  status     - Check the status of the scheduler"
    echo "  start      - Start the scheduler"
    echo "  stop       - Stop the scheduler"
    echo "  restart    - Restart the scheduler"
    echo "  logs       - View scheduler logs"
    echo "  enable     - Enable scheduler to start on boot"
    echo "  disable    - Disable scheduler from starting on boot"
    exit 1
}

# Check if command is provided
if [ $# -eq 0 ]; then
    show_usage
fi

COMMAND=$1

case $COMMAND in
    install)
        echo "Installing Vector Index Scheduler..."

        # Create systemd user directory if it doesn't exist
        mkdir -p "$SYSTEMD_DIR"

        # Update service file with actual paths
        PYTHON_PATH=$(which python)
        USER_NAME=$(whoami)

        # Create a customized service file
        cat > "$SYSTEMD_DIR/vector-index-scheduler.service" << EOF
[Unit]
Description=Vector Index Auto-Update Scheduler
After=network.target

[Service]
Type=simple
WorkingDirectory=$SCRIPT_DIR
Environment="PATH=$PATH"
ExecStart=$PYTHON_PATH $SCRIPT_DIR/vector_index_scheduler.py --config $SCRIPT_DIR/scheduler_config.yaml
Restart=on-failure
RestartSec=60
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vector-index-scheduler

[Install]
WantedBy=default.target
EOF

        # Reload systemd
        systemctl --user daemon-reload

        echo "Service installed successfully!"
        echo ""
        echo "Next steps:"
        echo "  1. Start the service: $0 start"
        echo "  2. Enable on boot: $0 enable"
        echo "  3. Check status: $0 status"
        ;;

    uninstall)
        echo "Uninstalling Vector Index Scheduler..."

        # Stop and disable the service
        systemctl --user stop vector-index-scheduler.service 2>/dev/null || true
        systemctl --user disable vector-index-scheduler.service 2>/dev/null || true

        # Remove service file
        rm -f "$SYSTEMD_DIR/vector-index-scheduler.service"

        # Reload systemd
        systemctl --user daemon-reload

        echo "Service uninstalled successfully!"
        ;;

    status)
        systemctl --user status vector-index-scheduler.service
        ;;

    start)
        echo "Starting Vector Index Scheduler..."
        systemctl --user start vector-index-scheduler.service
        echo "Service started!"
        echo "Check status with: $0 status"
        ;;

    stop)
        echo "Stopping Vector Index Scheduler..."
        systemctl --user stop vector-index-scheduler.service
        echo "Service stopped!"
        ;;

    restart)
        echo "Restarting Vector Index Scheduler..."
        systemctl --user restart vector-index-scheduler.service
        echo "Service restarted!"
        ;;

    logs)
        echo "Showing scheduler logs (Ctrl+C to exit)..."
        journalctl --user -u vector-index-scheduler.service -f
        ;;

    enable)
        echo "Enabling scheduler to start on boot..."
        systemctl --user enable vector-index-scheduler.service
        echo "Scheduler will now start automatically on boot!"
        ;;

    disable)
        echo "Disabling scheduler from starting on boot..."
        systemctl --user disable vector-index-scheduler.service
        echo "Scheduler will no longer start automatically on boot!"
        ;;

    *)
        echo "Error: Unknown command '$COMMAND'"
        echo ""
        show_usage
        ;;
esac
