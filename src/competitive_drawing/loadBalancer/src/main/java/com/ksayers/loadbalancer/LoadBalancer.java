package com.ksayers.loadbalancer;

import java.util.ArrayList;
import java.io.IOException;
import java.net.InetSocketAddress;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;


public class LoadBalancer 
{
    InetSocketAddress address = new InetSocketAddress("localhost", 8000);
    ArrayList<InetSocketAddress> serverAddresses = new ArrayList<InetSocketAddress>();
    HttpServer lbServer;
    Strategy strategy;

    public LoadBalancer(Strategy _strategy) throws IOException {
        strategy = _strategy;
        lbServer = HttpServer.create(address, 5);
        lbServer.createContext("/register", new RegisterHandler());
    }

    private class RegisterHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange httpExchange) {
            InetSocketAddress serverAddress = new InetSocketAddress("localhost", 8001);
            strategy.registerServer(new Server(serverAddress));
        }
    }
    
    public static void main( String[] args )
    {
        System.out.println( "Hello World!" );
    }
}
