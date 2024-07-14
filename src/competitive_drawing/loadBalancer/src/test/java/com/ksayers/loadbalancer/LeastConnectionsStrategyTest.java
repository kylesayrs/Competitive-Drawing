package com.ksayers.loadbalancer;

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
        return new TestSuite(LeastConnectionsStrategy.class);
    }

    public void testInitialization() {
        LeastConnectionsStrategy strategy = new LeastConnectionsStrategy();
        assertTrue(true);
    }
}
