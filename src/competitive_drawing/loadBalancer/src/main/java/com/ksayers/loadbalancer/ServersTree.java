package com.ksayers.loadbalancer;

import java.util.HashSet;
import java.util.TreeMap;
import java.util.logging.Logger;

public class ServersTree {
    static final Logger logger = Logger.getLogger(LeastConnectionsStrategy.class.getName());

    private final TreeMap<Integer, HashSet<Server>> numConnectionsToServers = new TreeMap<>();

    public void remove(Server server) {
        logger.info("remove");

        // remove server from set
        Integer numConnections = server.roomIds.size();
        HashSet<Server> serversWithSameNumConnections = numConnectionsToServers.get(numConnections);
        serversWithSameNumConnections.remove(server);

        // potentially remove set from tree
        if (serversWithSameNumConnections.size() <= 0) {
            numConnectionsToServers.remove(numConnections);
        }

        logger.info(String.format("%s", numConnectionsToServers));
    }

    public void put(Server server) {
        logger.info("put");
        Integer numConnections = server.roomIds.size();

        // insert server into set
        HashSet<Server> serversWithSameNumConnections = numConnectionsToServers.get(numConnections);
        if (serversWithSameNumConnections == null) {
            // insert server into set
            serversWithSameNumConnections = new HashSet<>();
            serversWithSameNumConnections.add(server);

            // insert set into tree
            numConnectionsToServers.put(numConnections, serversWithSameNumConnections);
        } else {
            // insert server into set
            serversWithSameNumConnections.add(server);
            
            // insert set into tree
            numConnectionsToServers.replace(numConnections, serversWithSameNumConnections);
        }


        logger.info(String.format("%s", numConnectionsToServers));
    }

    public boolean isEmpty() {
        return numConnectionsToServers.isEmpty();
    }

    public Integer firstKey() {
        return numConnectionsToServers.firstKey();
    }

    public HashSet<Server> get(Integer key) {
        return numConnectionsToServers.get(key);
    }
}
