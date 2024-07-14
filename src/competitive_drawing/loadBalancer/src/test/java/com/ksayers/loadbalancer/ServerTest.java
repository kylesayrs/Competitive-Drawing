package com.ksayers.loadbalancer;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

public class ServerTest 
    extends TestCase
{
    public ServerTest(String testName) {
        super(testName);
    }

    public static Test suite() {
        return new TestSuite(ServersTreeTest.class);
    }

    public void test() {
        Server server1 = new Server("localhost", 8000);
        Server server2 = new Server("localhost", 8000);
        Server server3 = new Server("localhost", 8001);

        // identity
        assertTrue(server1.hashCode() == server1.hashCode());
        assertTrue(server1 == server1);

        // deep copy
        assertTrue(server1.hashCode() == server2.hashCode());
        assertTrue(server1 == server2);

        // other
        assertTrue(server1.hashCode() != server3.hashCode());
        assertTrue(server1 != server3);
    }
}
