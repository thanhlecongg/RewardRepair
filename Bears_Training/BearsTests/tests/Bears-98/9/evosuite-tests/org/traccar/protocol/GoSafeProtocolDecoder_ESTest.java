/*
 * This file was automatically generated by EvoSuite
 * Tue Dec 10 04:34:30 GMT 2019
 */

package org.traccar.protocol;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.net.InetAddress;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.net.MockInetAddress;
import org.evosuite.runtime.mock.java.net.MockInetSocketAddress;
import org.jboss.netty.channel.Channel;
import org.jboss.netty.channel.DefaultChannelPipeline;
import org.jboss.netty.channel.local.DefaultLocalClientChannelFactory;
import org.jboss.netty.channel.local.DefaultLocalServerChannelFactory;
import org.jboss.netty.channel.local.LocalChannel;
import org.jboss.netty.channel.local.LocalServerChannel;
import org.junit.runner.RunWith;
import org.traccar.protocol.GoSafeProtocol;
import org.traccar.protocol.GoSafeProtocolDecoder;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true, useJEE = true) 
public class GoSafeProtocolDecoder_ESTest extends GoSafeProtocolDecoder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      GoSafeProtocol goSafeProtocol0 = new GoSafeProtocol();
      GoSafeProtocolDecoder goSafeProtocolDecoder0 = new GoSafeProtocolDecoder(goSafeProtocol0);
      DefaultLocalClientChannelFactory defaultLocalClientChannelFactory0 = new DefaultLocalClientChannelFactory();
      DefaultChannelPipeline defaultChannelPipeline0 = new DefaultChannelPipeline();
      LocalChannel localChannel0 = defaultLocalClientChannelFactory0.newChannel(defaultChannelPipeline0);
      InetAddress inetAddress0 = MockInetAddress.getLocalHost();
      MockInetSocketAddress mockInetSocketAddress0 = new MockInetSocketAddress(inetAddress0, 0);
      try { 
        goSafeProtocolDecoder0.decode(localChannel0, mockInetSocketAddress0, (Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      GoSafeProtocol goSafeProtocol0 = new GoSafeProtocol();
      GoSafeProtocolDecoder goSafeProtocolDecoder0 = new GoSafeProtocolDecoder(goSafeProtocol0);
      DefaultLocalServerChannelFactory defaultLocalServerChannelFactory0 = new DefaultLocalServerChannelFactory();
      DefaultChannelPipeline defaultChannelPipeline0 = new DefaultChannelPipeline();
      LocalServerChannel localServerChannel0 = defaultLocalServerChannelFactory0.newChannel(defaultChannelPipeline0);
      MockInetSocketAddress mockInetSocketAddress0 = new MockInetSocketAddress(1090);
      goSafeProtocolDecoder0.decode(localServerChannel0, mockInetSocketAddress0, "");
      assertEquals(0, (int)localServerChannel0.getId());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      GoSafeProtocol goSafeProtocol0 = new GoSafeProtocol();
      GoSafeProtocolDecoder goSafeProtocolDecoder0 = new GoSafeProtocolDecoder(goSafeProtocol0);
      MockInetSocketAddress mockInetSocketAddress0 = new MockInetSocketAddress(511);
      Integer integer0 = new Integer(0);
      try { 
        goSafeProtocolDecoder0.decode((Channel) null, mockInetSocketAddress0, integer0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // java.lang.Integer cannot be cast to java.lang.String
         //
         verifyException("org.traccar.protocol.GoSafeProtocolDecoder", e);
      }
  }
}
