/*
 * This file was automatically generated by EvoSuite
 * Tue Dec 10 03:52:29 GMT 2019
 */

package org.traccar.protocol;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jboss.netty.buffer.BigEndianHeapChannelBuffer;
import org.jboss.netty.buffer.ChannelBuffer;
import org.jboss.netty.channel.ChannelHandlerContext;
import org.jboss.netty.channel.DefaultChannelPipeline;
import org.jboss.netty.channel.local.DefaultLocalServerChannelFactory;
import org.jboss.netty.channel.local.LocalServerChannel;
import org.junit.runner.RunWith;
import org.traccar.protocol.TmgFrameDecoder;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true, useJEE = true) 
public class TmgFrameDecoder_ESTest extends TmgFrameDecoder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      TmgFrameDecoder tmgFrameDecoder0 = new TmgFrameDecoder();
      DefaultChannelPipeline defaultChannelPipeline0 = new DefaultChannelPipeline();
      DefaultLocalServerChannelFactory defaultLocalServerChannelFactory0 = new DefaultLocalServerChannelFactory();
      LocalServerChannel localServerChannel0 = defaultLocalServerChannelFactory0.newChannel(defaultChannelPipeline0);
      byte[] byteArray0 = new byte[7];
      byteArray0[0] = (byte)109;
      BigEndianHeapChannelBuffer bigEndianHeapChannelBuffer0 = new BigEndianHeapChannelBuffer(byteArray0);
      tmgFrameDecoder0.decode((ChannelHandlerContext) null, localServerChannel0, bigEndianHeapChannelBuffer0);
      //  // Unstable assertion: assertEquals(1, (int)localServerChannel0.getId());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      TmgFrameDecoder tmgFrameDecoder0 = new TmgFrameDecoder();
      DefaultChannelPipeline defaultChannelPipeline0 = new DefaultChannelPipeline();
      DefaultLocalServerChannelFactory defaultLocalServerChannelFactory0 = new DefaultLocalServerChannelFactory();
      LocalServerChannel localServerChannel0 = defaultLocalServerChannelFactory0.newChannel(defaultChannelPipeline0);
      try { 
        tmgFrameDecoder0.decode((ChannelHandlerContext) null, localServerChannel0, (ChannelBuffer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.traccar.protocol.TmgFrameDecoder", e);
      }
  }
}
