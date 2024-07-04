package com.ksayers.loadbalancer;


public interface Strategy {
    void registerServer(Server server);
    void deregisterServer(Server server);
    Server selectServer();
}
