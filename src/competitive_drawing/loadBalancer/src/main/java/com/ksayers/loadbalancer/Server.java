package com.ksayers.loadbalancer;

import java.net.InetSocketAddress;
import java.util.HashSet;

public class Server {
    public InetSocketAddress address;
    public HashSet<String> roomIds = new HashSet<>();

    public Server(InetSocketAddress _address) {
        address = _address;
    }

    public Server(String hostname, int port) {
        this(new InetSocketAddress(hostname, port));
    }

    public final int numConnections() {
        return roomIds.size();
    }

    @Override
    public final int hashCode() {
        return address.hashCode();
    }

    @Override
    public final boolean equals(Object other) {
        return other instanceof Server && hashCode() == other.hashCode();
    }
}
