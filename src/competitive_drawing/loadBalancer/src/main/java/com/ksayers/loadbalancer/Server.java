package com.ksayers.loadbalancer;

import java.net.InetSocketAddress;
import java.util.HashSet;

public class Server {
    public InetSocketAddress address;
    public HashSet<String> roomIds = new HashSet<String>();

    public Server(InetSocketAddress _address) {
        address = _address;
    }
}
