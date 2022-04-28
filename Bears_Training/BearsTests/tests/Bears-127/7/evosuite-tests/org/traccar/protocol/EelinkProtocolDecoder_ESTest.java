/*
 * This file was automatically generated by EvoSuite
 * Tue Dec 10 03:36:31 GMT 2019
 */

package org.traccar.protocol;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.cloudhopper.commons.util.NamingThreadFactory;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.LinkedList;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.net.MockInetSocketAddress;
import org.jboss.netty.buffer.ByteBufferBackedChannelBuffer;
import org.jboss.netty.buffer.ChannelBuffer;
import org.jboss.netty.buffer.CompositeChannelBuffer;
import org.jboss.netty.buffer.DuplicatedChannelBuffer;
import org.jboss.netty.buffer.HeapChannelBufferFactory;
import org.jboss.netty.channel.Channel;
import org.jboss.netty.channel.DefaultChannelPipeline;
import org.jboss.netty.channel.local.DefaultLocalClientChannelFactory;
import org.jboss.netty.channel.local.DefaultLocalServerChannelFactory;
import org.jboss.netty.channel.local.LocalChannel;
import org.jboss.netty.channel.local.LocalServerChannel;
import org.jboss.netty.channel.socket.nio.NioWorker;
import org.junit.runner.RunWith;
import org.traccar.protocol.EelinkProtocol;
import org.traccar.protocol.EelinkProtocolDecoder;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true, useJEE = true) 
public class EelinkProtocolDecoder_ESTest extends EelinkProtocolDecoder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      EelinkProtocol eelinkProtocol0 = new EelinkProtocol();
      EelinkProtocolDecoder eelinkProtocolDecoder0 = new EelinkProtocolDecoder(eelinkProtocol0);
      DefaultLocalClientChannelFactory defaultLocalClientChannelFactory0 = new DefaultLocalClientChannelFactory();
      DefaultChannelPipeline defaultChannelPipeline0 = new DefaultChannelPipeline();
      LocalChannel localChannel0 = defaultLocalClientChannelFactory0.newChannel(defaultChannelPipeline0);
      InetSocketAddress inetSocketAddress0 = MockInetSocketAddress.createUnresolved("I+n;u", 6);
      LinkedList<ChannelBuffer> linkedList0 = new LinkedList<ChannelBuffer>();
      CompositeChannelBuffer compositeChannelBuffer0 = new CompositeChannelBuffer((ByteOrder) null, linkedList0, true);
      try { 
        eelinkProtocolDecoder0.decode(localChannel0, inetSocketAddress0, compositeChannelBuffer0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Readable bytes exceeded - Need 2, maximum is 0
         //
         verifyException("org.jboss.netty.buffer.AbstractChannelBuffer", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      EelinkProtocol eelinkProtocol0 = new EelinkProtocol();
      EelinkProtocolDecoder eelinkProtocolDecoder0 = new EelinkProtocolDecoder(eelinkProtocol0);
      DefaultLocalServerChannelFactory defaultLocalServerChannelFactory0 = new DefaultLocalServerChannelFactory();
      DefaultChannelPipeline defaultChannelPipeline0 = new DefaultChannelPipeline();
      LocalServerChannel localServerChannel0 = defaultLocalServerChannelFactory0.newChannel(defaultChannelPipeline0);
      MockInetSocketAddress mockInetSocketAddress0 = new MockInetSocketAddress("", 20);
      NamingThreadFactory namingThreadFactory0 = new NamingThreadFactory("`Ny5`\"Is+gh3%{pG", false);
      ThreadPoolExecutor.AbortPolicy threadPoolExecutor_AbortPolicy0 = new ThreadPoolExecutor.AbortPolicy();
      ScheduledThreadPoolExecutor scheduledThreadPoolExecutor0 = new ScheduledThreadPoolExecutor(20, namingThreadFactory0, threadPoolExecutor_AbortPolicy0);
      NioWorker nioWorker0 = new NioWorker(scheduledThreadPoolExecutor0);
      TimeUnit timeUnit0 = TimeUnit.SECONDS;
      ScheduledFuture<?> scheduledFuture0 = scheduledThreadPoolExecutor0.scheduleWithFixedDelay(nioWorker0, 23, 18, timeUnit0);
      try { 
        eelinkProtocolDecoder0.decode(localServerChannel0, mockInetSocketAddress0, scheduledFuture0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // java.util.concurrent.ScheduledThreadPoolExecutor$ScheduledFutureTask cannot be cast to org.jboss.netty.buffer.ChannelBuffer
         //
         verifyException("org.traccar.protocol.EelinkProtocolDecoder", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      EelinkProtocol eelinkProtocol0 = new EelinkProtocol();
      EelinkProtocolDecoder eelinkProtocolDecoder0 = new EelinkProtocolDecoder(eelinkProtocol0);
      DefaultLocalClientChannelFactory defaultLocalClientChannelFactory0 = new DefaultLocalClientChannelFactory();
      DefaultChannelPipeline defaultChannelPipeline0 = new DefaultChannelPipeline();
      LocalChannel localChannel0 = defaultLocalClientChannelFactory0.newChannel(defaultChannelPipeline0);
      InetSocketAddress inetSocketAddress0 = MockInetSocketAddress.createUnresolved("y&awHW2a1s{};0", 0);
      ByteOrder byteOrder0 = ByteOrder.BIG_ENDIAN;
      HeapChannelBufferFactory heapChannelBufferFactory0 = new HeapChannelBufferFactory(byteOrder0);
      ByteBuffer byteBuffer0 = ByteBuffer.allocateDirect(129);
      ChannelBuffer channelBuffer0 = heapChannelBufferFactory0.getBuffer(byteBuffer0);
      DuplicatedChannelBuffer duplicatedChannelBuffer0 = new DuplicatedChannelBuffer(channelBuffer0);
      try { 
        eelinkProtocolDecoder0.decode(localChannel0, inetSocketAddress0, duplicatedChannelBuffer0);
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
      MockInetSocketAddress mockInetSocketAddress0 = new MockInetSocketAddress("y@]Y^XHH/m%Mk>L*c", 5);
      ByteBuffer byteBuffer0 = ByteBuffer.allocateDirect(2154);
      ByteBufferBackedChannelBuffer byteBufferBackedChannelBuffer0 = new ByteBufferBackedChannelBuffer(byteBuffer0);
      DuplicatedChannelBuffer duplicatedChannelBuffer0 = new DuplicatedChannelBuffer(byteBufferBackedChannelBuffer0);
      try { 
        eelinkProtocolDecoder0.decode((Channel) null, mockInetSocketAddress0, duplicatedChannelBuffer0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.traccar.BaseProtocolDecoder", e);
      }
  }
}
