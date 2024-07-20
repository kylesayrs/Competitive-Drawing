package com.ksayers.loadbalancer;

import java.net.InetSocketAddress;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

public class LeastConnectionsStrategyTest 
    extends TestCase
{
    public LeastConnectionsStrategyTest(String testName) {
        super(testName);
    }

    public static Test suite() {
        return new TestSuite(LeastConnectionsStrategyTest.class);
    }

    public void testInitialization() {
        LeastConnectionsStrategy strategy = new LeastConnectionsStrategy();
        
        InetSocketAddress address1 = new InetSocketAddress("localhost", 8000);

        strategy.addServer(address1);
    }
}
