package com.ksayers.loadbalancer;

import java.net.InetSocketAddress;
import java.util.HashMap;
import java.util.HashSet;
import java.util.logging.Logger;


public class LeastConnectionsStrategy implements Strategy {
    static final Logger logger = Logger.getLogger(LeastConnectionsStrategy.class.getName());

    HashMap<InetSocketAddress, Server> addressToServer = new HashMap<>();
    HashMap<String, Server> roomIdToServer = new HashMap<>();
    ServersTree serversTree = new ServersTree();


    @Override
    public void addServer(InetSocketAddress address) {
        // create new server
        Server newServer = new Server(address);
        addressToServer.put(address, newServer);

        // add to address lookup
        addressToServer.put(newServer.address, newServer);

        // insert into tree
        serversTree.put(newServer);
    }

    @Override
    public void removeServer(InetSocketAddress address) {
        Server server = addressToServer.get(address);

        // remove room id mappings
        for (String roomId : server.roomIds) {
            roomIdToServer.remove(roomId);
        }

        // remove from address lookup
        addressToServer.remove(server.address);

        // remove from tree
        serversTree.remove(server);
    }

    @Override
    public void endSession(String roomId) {
        Server server = roomIdToServer.get(roomId);
        
        // remove from lookup
        roomIdToServer.remove(roomId);

        // remove server from tree
        serversTree.remove(server);
        server.roomIds.remove(roomId);
        serversTree.put(server);
    }


    @Override
    public InetSocketAddress selectServer(String roomId) {
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
        serversTree.put(server);

        return server.address;
    }
}
