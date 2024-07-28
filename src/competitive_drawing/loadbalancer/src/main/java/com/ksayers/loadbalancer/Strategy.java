package com.ksayers.loadbalancer;

import java.net.InetSocketAddress;


public interface Strategy {
    boolean addServer(InetSocketAddress address);
    boolean removeServer(InetSocketAddress address);
    boolean endSession(String roomId);
    InetSocketAddress selectServer(String roomId);
}
