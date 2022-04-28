/*
 * This file was automatically generated by EvoSuite
 * Tue Dec 10 03:32:50 GMT 2019
 */

package org.traccar.protocol;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.net.MockInetSocketAddress;
import org.jboss.netty.buffer.BigEndianHeapChannelBuffer;
import org.jboss.netty.buffer.LittleEndianHeapChannelBuffer;
import org.jboss.netty.channel.Channel;
import org.jboss.netty.channel.DefaultChannelPipeline;
import org.jboss.netty.channel.local.DefaultLocalClientChannelFactory;
import org.jboss.netty.channel.local.LocalChannel;
import org.junit.runner.RunWith;
import org.traccar.protocol.EelinkProtocol;
import org.traccar.protocol.EelinkProtocolDecoder;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true, useJEE = true) 
public class EelinkProtocolDecoder_ESTest extends EelinkProtocolDecoder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      EelinkProtocol eelinkProtocol0 = new EelinkProtocol();
      EelinkProtocolDecoder eelinkProtocolDecoder0 = new EelinkProtocolDecoder(eelinkProtocol0);
      byte[] byteArray0 = new byte[9];
      byteArray0[2] = (byte)65;
      BigEndianHeapChannelBuffer bigEndianHeapChannelBuffer0 = new BigEndianHeapChannelBuffer(byteArray0);
      try { 
        eelinkProtocolDecoder0.decode((Channel) null, (SocketAddress) null, bigEndianHeapChannelBuffer0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.traccar.BaseProtocolDecoder", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      EelinkProtocolDecoder eelinkProtocolDecoder0 = new EelinkProtocolDecoder((EelinkProtocol) null);
      DefaultChannelPipeline defaultChannelPipeline0 = new DefaultChannelPipeline();
      InetSocketAddress inetSocketAddress0 = MockInetSocketAddress.createUnresolved("vibration", 126);
      try { 
        eelinkProtocolDecoder0.decode((Channel) null, inetSocketAddress0, defaultChannelPipeline0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // org.jboss.netty.channel.DefaultChannelPipeline cannot be cast to org.jboss.netty.buffer.ChannelBuffer
         //
         verifyException("org.traccar.protocol.EelinkProtocolDecoder", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      EelinkProtocolDecoder eelinkProtocolDecoder0 = new EelinkProtocolDecoder((EelinkProtocol) null);
      MockInetSocketAddress mockInetSocketAddress0 = new MockInetSocketAddress(172);
      byte[] byteArray0 = new byte[7];
      LittleEndianHeapChannelBuffer littleEndianHeapChannelBuffer0 = new LittleEndianHeapChannelBuffer(byteArray0);
      try { 
        eelinkProtocolDecoder0.decode((Channel) null, mockInetSocketAddress0, littleEndianHeapChannelBuffer0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.traccar.BaseProtocolDecoder", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      EelinkProtocol eelinkProtocol0 = new EelinkProtocol();
      EelinkProtocolDecoder eelinkProtocolDecoder0 = new EelinkProtocolDecoder(eelinkProtocol0);
      DefaultLocalClientChannelFactory defaultLocalClientChannelFactory0 = new DefaultLocalClientChannelFactory();
      DefaultChannelPipeline defaultChannelPipeline0 = new DefaultChannelPipeline();
      LocalChannel localChannel0 = defaultLocalClientChannelFactory0.newChannel(defaultChannelPipeline0);
      InetSocketAddress inetSocketAddress0 = MockInetSocketAddress.createUnresolved("", 24);
      byte[] byteArray0 = new byte[5];
      LittleEndianHeapChannelBuffer littleEndianHeapChannelBuffer0 = new LittleEndianHeapChannelBuffer(byteArray0);
      try { 
        eelinkProtocolDecoder0.decode(localChannel0, inetSocketAddress0, littleEndianHeapChannelBuffer0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Not enough readable bytes - Need 2, maximum is 0
         //
         verifyException("org.jboss.netty.buffer.AbstractChannelBuffer", e);
      }
  }
}
