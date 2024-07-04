package com.ksayers.loadbalancer;

import java.net.InetSocketAddress;

public class Server {
    public InetSocketAddress address;

    public Server(InetSocketAddress _address) {
        address = _address;
    }
}
