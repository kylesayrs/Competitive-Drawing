package com.ksayers.loadbalancer;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.logging.Logger;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;


public class LoadBalancer 
{
    static final Logger logger = Logger.getLogger(LeastConnectionsStrategy.class.getName());

    InetSocketAddress address = new InetSocketAddress("localhost", 8000);
    HttpServer server;
    Strategy strategy;
    ThreadPoolExecutor serverThreadPool = (ThreadPoolExecutor) Executors.newFixedThreadPool(10);

    public LoadBalancer(Strategy _strategy) throws IOException {
        strategy = _strategy;
        server = HttpServer.create(address, 5);
        server.setExecutor(serverThreadPool);
        server.createContext("/register", new RegisterHandler());
        server.createContext("/unregister", new UnregisterHandler());
        server.createContext("/", new RequestHandler());
    }

    private class RegisterHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange httpExchange) {
            InetSocketAddress serverAddress = new InetSocketAddress("localhost", 8001);
            strategy.addServer(serverAddress);
        }
    }

    private class UnregisterHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange httpExchange) {
            InetSocketAddress serverAddress = new InetSocketAddress("localhost", 8001);
            strategy.removeServer(serverAddress);
        }
    }

    private class RequestHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange httpExchange) {
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
            /*

            // send proxy
            try {
                URL url = new URI(String.format(
                    "http://%s:%s/%s",
                    address.getHostName(),
                    address.getPort(),
                    httpExchange.getRequestURI()
                )).toURL();
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setConnectTimeout(1000);
                connection.setRequestMethod(httpExchange.getRequestMethod());
                connection.setDoOutput(true);
                
                // copy data
                OutputStream outputStream = connection.getOutputStream();
                IOUtils.copy(httpExchange.getRequestBody(), outputStream);
                outputStream.flush();
                outputStream.close();

                connection.getInputStream();
                
                // check response code
                int responseCode = connection.getResponseCode();
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    System.out.println(String.format("Successfully sent %s message to %s at %s", path, nodeId, url));
                } else {
                    System.err.println(String.format("Failed to send %s message node %s at %s", path, nodeId, url));
                    return new Pair<Integer,String>(responseCode, null);
                }

                // get response body
                InputStream inputStream = connection.getInputStream();
                byte[] responseBodyBytes = inputStream.readAllBytes();
                String responseBody = new String(responseBodyBytes);

                httpExchange.response
                
                connection.disconnect();

                return new Pair<Integer,String>(responseCode, responseBody);

            } catch (Exception exception) {
                System.err.println(String.format("Failed to send %s message node %s", path, nodeId));

                return new Pair<Integer,String>(null, null);
            }
            
            // forward request to server
             */
        }
    }
    
    public static void main( String[] args )
    {
        System.out.println( "Hello World!" );
        InetSocketAddress a = new InetSocketAddress("localhost", 8000);
        InetSocketAddress b = new InetSocketAddress("localhost", 8000);

        System.out.println(a.hashCode() == b.hashCode());
    }
}
