nohup bash -c "/docker-entrypoint.sh postgres &" && \
sleep 3 && \
jina pod --uses config.yml $@
