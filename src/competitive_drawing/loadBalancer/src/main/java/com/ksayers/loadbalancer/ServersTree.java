package com.ksayers.loadbalancer;

import java.util.HashSet;
import java.util.TreeMap;

public class ServersTree {
    TreeMap<Integer, HashSet<Server>> numConnectionsToServers = new TreeMap<>();

    public void remove(Server server) {
        // remove server from set
        Integer numConnections = server.roomIds.size();
        HashSet<Server> serversWithSameNumConnections = numConnectionsToServers.get(numConnections);
        serversWithSameNumConnections.remove(server);

        // potentially remove set from tree
        if (serversWithSameNumConnections.size() <= 0) {
            numConnectionsToServers.remove(numConnections);
        }
    }


    public void put(Server server) {
        Integer numConnections = server.roomIds.size();

        // insert server into set
        HashSet<Server> serversWithSameNumConnections = numConnectionsToServers.get(numConnections);
        if (serversWithSameNumConnections == null) {
            serversWithSameNumConnections = new HashSet<>();
        }
        serversWithSameNumConnections.add(server);

        // insert set into tree
        numConnectionsToServers.replace(numConnections, serversWithSameNumConnections);
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
