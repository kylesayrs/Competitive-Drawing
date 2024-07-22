package com.ksayers.loadbalancer;

import java.net.InetSocketAddress;
import java.util.HashMap;
import java.util.HashSet;
import java.util.NoSuchElementException;
import java.util.logging.Logger;


public class LeastConnectionsStrategy implements Strategy {
    static final Logger logger = Logger.getLogger(LeastConnectionsStrategy.class.getName());

    HashMap<InetSocketAddress, Server> addressToServer = new HashMap<>();
    HashMap<String, Server> roomIdToServer = new HashMap<>();
    ServersTree serversTree = new ServersTree();

    @Override
    public boolean addServer(InetSocketAddress address) {
        if (addressToServer.containsKey(address)) {
            return false;
        }

        // create new server
        Server newServer = new Server(address);

        // add to address lookup
        addressToServer.put(address, newServer);

        // insert into tree
        serversTree.put(newServer.numConnections(), newServer);

        return true;
    }

    @Override
    public boolean removeServer(InetSocketAddress address) {
        Server server = addressToServer.get(address);
        if (server == null) {
            return false;
        }

        // remove room id mappings
        for (String roomId : server.roomIds) {
            if (roomIdToServer.remove(roomId) == null) {
                logger.warning("Attempted to remove room but no such room was found");
            }
        }

        // remove from address lookup
        if (addressToServer.remove(server.address) == null) {
            throw new NoSuchElementException("Address not found in address server table");
        }

        // remove from tree
        serversTree.remove(server);
        return true;
    }

    @Override
    public boolean endSession(String roomId) {
        Server server = roomIdToServer.get(roomId);
        if (server == null) {
            return false;
        }
        
        // remove from lookup
        if (roomIdToServer.remove(roomId) == null) {
            throw new NoSuchElementException("Session not found in ");
        }

        // remove server from tree
        serversTree.remove(server);
        server.roomIds.remove(roomId);
        serversTree.put(server.numConnections(), server);

        return true;
    }


    @Override
    public InetSocketAddress selectServer(String roomId) {
        // check lookup
        Server server = roomIdToServer.get(roomId);
        if (server != null) {
            return server.address;
        }

        if (serversTree.isEmpty()) {
            logger.warning(String.format("Could not assign roomId %s, no servers are available", roomId));
            return null;
        }

        // retrieve any least connections server
        Integer leastNumConnections = serversTree.firstKey();
        HashSet<Server> leastConnectionsServers = serversTree.get(leastNumConnections);
        assert !leastConnectionsServers.isEmpty();
        server = leastConnectionsServers.iterator().next();
        
        // add to lookup
        roomIdToServer.put(roomId, server);
        
        // update tree
        serversTree.remove(server);
        server.roomIds.add(roomId);
        serversTree.put(server.numConnections(), server);

        return server.address;
    }
}
