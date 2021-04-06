bash /docker-entrypoint.sh postgres &
sleep 1 &
jina pod --uses config.yml