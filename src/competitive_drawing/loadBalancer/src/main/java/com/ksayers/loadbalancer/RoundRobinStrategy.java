package com.ksayers.loadbalancer;

import java.util.ArrayList;
import java.util.logging.Logger;


public class RoundRobinStrategy implements Strategy {
    ArrayList<Server> servers = new ArrayList<Server>();
    static final Logger logger = Logger.getLogger(RoundRobinStrategy.class.getName());
    Integer index = 0;

    public void registerServer(Server server) {
        servers.add(server);
    }

    public void deregisterServer(Server server) {
        if (!servers.remove(server)) {
            logger.warning(String.format("Failed to deregsiter server %s", server.address));
        }
    }

    public Server selectServer() {
        int num_servers = servers.size();
        Server selectedServer = servers.get(index % num_servers);
        index = (index + 1) % num_servers;
        return selectedServer;
    }
}
