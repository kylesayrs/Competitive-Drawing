package com.ksayers.loadbalancer;

import java.net.InetSocketAddress;


public interface Strategy {
    void addServer(InetSocketAddress address);
    void removeServer(InetSocketAddress address);
    void endSession(String roomId);
    InetSocketAddress selectServer(String roomId);
}
