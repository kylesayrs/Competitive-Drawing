package com.ksayers.loadbalancer;

import java.util.HashSet;
import java.util.NoSuchElementException;
import java.util.TreeMap;
import java.util.logging.Logger;

public class ServersTree {
    static final Logger logger = Logger.getLogger(ServersTree.class.getName());

    private final TreeMap<Integer, HashSet<Server>> numConnectionsToServers = new TreeMap<>();

    public void remove(Server server) {
        // find set
        Integer numConnections = server.roomIds.size();
        HashSet<Server> serversWithSameNumConnections = numConnectionsToServers.get(numConnections);
        if (serversWithSameNumConnections == null) {
            throw new NoSuchElementException();
        }
        
        // remove server from set
        if (!serversWithSameNumConnections.contains(server)) {
            throw new NoSuchElementException();
        }
        serversWithSameNumConnections.remove(server);

        // potentially remove set from tree
        if (serversWithSameNumConnections.size() <= 0) {
            numConnectionsToServers.remove(numConnections);
        }

        logger.info(String.format("%s", numConnectionsToServers));
    }

    public void put(Integer numConnections, Server server) {
        // insert server into set
        HashSet<Server> serversWithSameNumConnections = numConnectionsToServers.get(numConnections);
        if (serversWithSameNumConnections == null) {
            serversWithSameNumConnections = new HashSet<>();
        }

        // insert server into set
        serversWithSameNumConnections.add(server);
        numConnectionsToServers.put(numConnections, serversWithSameNumConnections);

        logger.info(String.format("%s", numConnectionsToServers));
    }

    public boolean isEmpty() {
        return numConnectionsToServers.isEmpty();
    }

    public int size() {
        int _size = 0;
        for (HashSet<Server> serversSet : numConnectionsToServers.values()) {
            _size += serversSet.size();
        }
        return _size;
    }

    public Integer firstKey() {
        return numConnectionsToServers.firstKey();
    }

    public HashSet<Server> get(Integer key) {
        return numConnectionsToServers.get(key);
    }
}
