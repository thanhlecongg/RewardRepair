/*
 * This file was automatically generated by EvoSuite
 * Tue Dec 10 02:01:31 GMT 2019
 */

package org.traccar.protocol;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.net.MockInetSocketAddress;
import org.jboss.netty.channel.DefaultChannelPipeline;
import org.jboss.netty.channel.local.DefaultLocalClientChannelFactory;
import org.jboss.netty.channel.local.DefaultLocalServerChannelFactory;
import org.jboss.netty.channel.local.LocalChannel;
import org.jboss.netty.channel.local.LocalServerChannel;
import org.junit.runner.RunWith;
import org.traccar.protocol.Gl200Protocol;
import org.traccar.protocol.Gl200ProtocolDecoder;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true, useJEE = true) 
public class Gl200ProtocolDecoder_ESTest extends Gl200ProtocolDecoder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Gl200Protocol gl200Protocol0 = new Gl200Protocol();
      Gl200ProtocolDecoder gl200ProtocolDecoder0 = new Gl200ProtocolDecoder(gl200Protocol0);
      DefaultLocalServerChannelFactory defaultLocalServerChannelFactory0 = new DefaultLocalServerChannelFactory();
      DefaultChannelPipeline defaultChannelPipeline0 = new DefaultChannelPipeline();
      LocalServerChannel localServerChannel0 = defaultLocalServerChannelFactory0.newChannel(defaultChannelPipeline0);
      MockInetSocketAddress mockInetSocketAddress0 = new MockInetSocketAddress(79);
      try { 
        gl200ProtocolDecoder0.decode(localServerChannel0, mockInetSocketAddress0, (Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.traccar.protocol.Gl200ProtocolDecoder", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Gl200Protocol gl200Protocol0 = new Gl200Protocol();
      Gl200ProtocolDecoder gl200ProtocolDecoder0 = new Gl200ProtocolDecoder(gl200Protocol0);
      DefaultLocalClientChannelFactory defaultLocalClientChannelFactory0 = new DefaultLocalClientChannelFactory();
      DefaultChannelPipeline defaultChannelPipeline0 = new DefaultChannelPipeline();
      LocalChannel localChannel0 = defaultLocalClientChannelFactory0.newChannel(defaultChannelPipeline0);
      MockInetSocketAddress mockInetSocketAddress0 = new MockInetSocketAddress(2842);
      try { 
        gl200ProtocolDecoder0.decode(localChannel0, mockInetSocketAddress0, defaultChannelPipeline0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // org.jboss.netty.channel.DefaultChannelPipeline cannot be cast to java.lang.String
         //
         verifyException("org.traccar.protocol.Gl200ProtocolDecoder", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Gl200Protocol gl200Protocol0 = new Gl200Protocol();
      Gl200ProtocolDecoder gl200ProtocolDecoder0 = new Gl200ProtocolDecoder(gl200Protocol0);
      DefaultLocalClientChannelFactory defaultLocalClientChannelFactory0 = new DefaultLocalClientChannelFactory();
      DefaultChannelPipeline defaultChannelPipeline0 = new DefaultChannelPipeline();
      LocalChannel localChannel0 = defaultLocalClientChannelFactory0.newChannel(defaultChannelPipeline0);
      MockInetSocketAddress mockInetSocketAddress0 = new MockInetSocketAddress("(ZBR%N2=@K", 310);
      gl200ProtocolDecoder0.decode(localChannel0, mockInetSocketAddress0, "(ZBR%N2=@K");
      assertEquals(0, (int)localChannel0.getId());
  }
}
