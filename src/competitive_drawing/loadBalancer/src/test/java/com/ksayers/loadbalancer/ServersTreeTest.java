package com.ksayers.loadbalancer;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

public class ServersTreeTest 
    extends TestCase
{
    public ServersTreeTest(String testName) {
        super(testName);
    }

    public static Test suite() {
        return new TestSuite(ServersTreeTest.class);
    }

    public void testInitialization() {
        ServersTree serversTree = new ServersTree();
        assertTrue(serversTree.isEmpty());
        assertTrue(serversTree.size() == 0);
        assertTrue(serversTree.get(0) == null);
    }

    public void testPut() {
        ServersTree serversTree = new ServersTree();

        Server server1 = new Server("localhost", 8000);
        Server server2 = new Server("localhost", 8001);
        Server server3 = new Server("localhost", 8002);

        // put
        serversTree.put(0, server1);
        assertTrue(serversTree.size() == 1);

        // put again
        serversTree.put(0, server2);
        assertTrue(serversTree.size() == 2);

        // put duplicate
        serversTree.put(100, server3);
        assertTrue(serversTree.size() == 3);
    }

    public void testDuplicatePut() {
        ServersTree serversTree = new ServersTree();

        Server server1 = new Server("localhost", 8000);
        Server server1Copy = new Server("localhost", 8000);
        Server server2 = new Server("localhost", 8001);

        // put
        serversTree.put(0, server1);
        assertTrue(serversTree.size() == 1);

        // put again
        serversTree.put(0, server1);
        assertTrue(serversTree.size() == 1);

        // put duplicate
        serversTree.put(0, server1Copy);
        assertTrue(serversTree.size() == 1);

        // put other
        serversTree.put(0, server2);
        assertTrue(serversTree.size() == 2);
    }
}
