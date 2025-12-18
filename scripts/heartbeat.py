# This file will constantly request the config file from the server (GET requests)
# This happens in a set time interval (not decided yet)
# It will replace the local config file if there are any changes
# If there are any changes, it will send the necessary systemd signals to start/stop services
# The heartbeat system will also monitor the status of the Jetson and report any issues (POST requests)
# The health report can actually be combined with the config file request to reduce the number of requests
