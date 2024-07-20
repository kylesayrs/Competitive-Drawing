package com.ksayers.loadbalancer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.Logger;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;


public class LoadBalancer
{
    static final Logger logger = Logger.getLogger(LoadBalancer.class.getName());

    private final InetSocketAddress address = new InetSocketAddress("localhost", 8000);
    private final HttpServer server = HttpServer.create(address, 5);
    private final LeastConnectionsStrategy strategy = new LeastConnectionsStrategy();
    private final ExecutorService serverThread = Executors.newSingleThreadExecutor();

    public LoadBalancer() throws IOException {
        server.setExecutor(serverThread);
        server.createContext("/", new RoutingHandler());
    }

    public void start() {
        server.start();
        System.out.println(String.format("Listening on %s", address));
    }

    private class RoutingHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange httpExchange) throws IOException {
            String path = httpExchange.getRequestURI().getPath();

            switch (path) {
                case "/register" -> handleRegister(httpExchange);
            
                case "/unregister" -> handleUnregister(httpExchange);

                case "/favicon.ico" -> {
                    try (OutputStream outputStream = httpExchange.getResponseBody()) {
                        httpExchange.sendResponseHeaders(200, 0);
                        outputStream.flush();
                    }
                }

            
                default -> handleRequest(httpExchange);
            }
        }

        private void handleRegister(HttpExchange httpExchange) throws IOException {
            InetSocketAddress serverAddress = new InetSocketAddress("localhost", 8001);
            strategy.addServer(serverAddress);

            try (OutputStream outputStream = httpExchange.getResponseBody()) {
                httpExchange.sendResponseHeaders(200, 0);
                outputStream.flush();
            }
        }

        private void handleUnregister(HttpExchange httpExchange) throws IOException {
            InetSocketAddress serverAddress = new InetSocketAddress("localhost", 8001);
            strategy.removeServer(serverAddress);

            try (OutputStream outputStream = httpExchange.getResponseBody()) {
                httpExchange.sendResponseHeaders(200, 0);
                outputStream.flush();
                outputStream.close();
            }
        }

        private void handleRequest(HttpExchange httpExchange) throws IOException {
            // extract room id
            Headers headers = httpExchange.getRequestHeaders();
            String roomId = headers.getFirst("Room-Id");
            if (roomId == null) {
                logger.warning(String.format("Request does not have Room-Id header"));
                return;
            }

            // select server
            InetSocketAddress address = strategy.selectServer(roomId);
            
            logger.info(String.format("Assigning %s to %s", roomId, address));
        }
    }
    
    public static void main( String[] args ) throws Exception {
        LoadBalancer loadBalancer = new LoadBalancer();
        loadBalancer.start();
    }
}
