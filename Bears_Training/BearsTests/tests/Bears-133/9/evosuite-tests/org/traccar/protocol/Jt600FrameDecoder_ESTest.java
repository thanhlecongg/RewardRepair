/*
 * This file was automatically generated by EvoSuite
 * Tue Dec 10 01:51:53 GMT 2019
 */

package org.traccar.protocol;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.Charset;
import java.text.ParseException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jboss.netty.buffer.ByteBufferBackedChannelBuffer;
import org.jboss.netty.buffer.ChannelBuffer;
import org.jboss.netty.buffer.DirectChannelBufferFactory;
import org.jboss.netty.channel.Channel;
import org.jboss.netty.channel.ChannelHandlerContext;
import org.jboss.netty.channel.DefaultChannelPipeline;
import org.jboss.netty.channel.local.DefaultLocalServerChannelFactory;
import org.jboss.netty.channel.local.LocalServerChannel;
import org.junit.runner.RunWith;
import org.traccar.protocol.Jt600FrameDecoder;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true, useJEE = true) 
public class Jt600FrameDecoder_ESTest extends Jt600FrameDecoder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Jt600FrameDecoder jt600FrameDecoder0 = new Jt600FrameDecoder();
      DefaultChannelPipeline defaultChannelPipeline0 = new DefaultChannelPipeline();
      DefaultLocalServerChannelFactory defaultLocalServerChannelFactory0 = new DefaultLocalServerChannelFactory();
      LocalServerChannel localServerChannel0 = defaultLocalServerChannelFactory0.newChannel(defaultChannelPipeline0);
      DirectChannelBufferFactory directChannelBufferFactory0 = new DirectChannelBufferFactory();
      Charset charset0 = Charset.defaultCharset();
      char[] charArray0 = new char[18];
      charArray0[0] = ',';
      CharBuffer charBuffer0 = CharBuffer.wrap(charArray0);
      ByteBuffer byteBuffer0 = charset0.encode(charBuffer0);
      ChannelBuffer channelBuffer0 = directChannelBufferFactory0.getBuffer(byteBuffer0);
      try { 
        jt600FrameDecoder0.decode((ChannelHandlerContext) null, localServerChannel0, channelBuffer0);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.traccar.protocol.Jt600FrameDecoder", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Jt600FrameDecoder jt600FrameDecoder0 = new Jt600FrameDecoder();
      DefaultChannelPipeline defaultChannelPipeline0 = new DefaultChannelPipeline();
      DefaultLocalServerChannelFactory defaultLocalServerChannelFactory0 = new DefaultLocalServerChannelFactory();
      LocalServerChannel localServerChannel0 = defaultLocalServerChannelFactory0.newChannel(defaultChannelPipeline0);
      DirectChannelBufferFactory directChannelBufferFactory0 = new DirectChannelBufferFactory(1024);
      Charset charset0 = Charset.defaultCharset();
      char[] charArray0 = new char[10];
      CharBuffer charBuffer0 = CharBuffer.wrap(charArray0);
      ByteBuffer byteBuffer0 = charset0.encode(charBuffer0);
      ChannelBuffer channelBuffer0 = directChannelBufferFactory0.getBuffer(byteBuffer0);
      try { 
        jt600FrameDecoder0.decode((ChannelHandlerContext) null, localServerChannel0, channelBuffer0);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.traccar.protocol.Jt600FrameDecoder", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Jt600FrameDecoder jt600FrameDecoder0 = new Jt600FrameDecoder();
      try { 
        jt600FrameDecoder0.decode((ChannelHandlerContext) null, (Channel) null, (ChannelBuffer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.traccar.protocol.Jt600FrameDecoder", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Jt600FrameDecoder jt600FrameDecoder0 = new Jt600FrameDecoder();
      DefaultChannelPipeline defaultChannelPipeline0 = new DefaultChannelPipeline();
      DefaultLocalServerChannelFactory defaultLocalServerChannelFactory0 = new DefaultLocalServerChannelFactory();
      LocalServerChannel localServerChannel0 = defaultLocalServerChannelFactory0.newChannel(defaultChannelPipeline0);
      DirectChannelBufferFactory directChannelBufferFactory0 = new DirectChannelBufferFactory();
      Charset charset0 = Charset.defaultCharset();
      char[] charArray0 = new char[18];
      charArray0[0] = '(';
      CharBuffer charBuffer0 = CharBuffer.wrap(charArray0);
      ByteBuffer byteBuffer0 = charset0.encode(charBuffer0);
      ChannelBuffer channelBuffer0 = directChannelBufferFactory0.getBuffer(byteBuffer0);
      jt600FrameDecoder0.decode((ChannelHandlerContext) null, localServerChannel0, channelBuffer0);
      //  // Unstable assertion: assertEquals(55, (int)localServerChannel0.getId());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Jt600FrameDecoder jt600FrameDecoder0 = new Jt600FrameDecoder();
      DefaultChannelPipeline defaultChannelPipeline0 = new DefaultChannelPipeline();
      DefaultLocalServerChannelFactory defaultLocalServerChannelFactory0 = new DefaultLocalServerChannelFactory();
      LocalServerChannel localServerChannel0 = defaultLocalServerChannelFactory0.newChannel(defaultChannelPipeline0);
      DirectChannelBufferFactory directChannelBufferFactory0 = new DirectChannelBufferFactory();
      Charset charset0 = Charset.defaultCharset();
      char[] charArray0 = new char[18];
      charArray0[0] = '(';
      charArray0[8] = ')';
      CharBuffer charBuffer0 = CharBuffer.wrap(charArray0);
      ByteBuffer byteBuffer0 = charset0.encode(charBuffer0);
      ChannelBuffer channelBuffer0 = directChannelBufferFactory0.getBuffer(byteBuffer0);
      ByteBufferBackedChannelBuffer byteBufferBackedChannelBuffer0 = (ByteBufferBackedChannelBuffer)jt600FrameDecoder0.decode((ChannelHandlerContext) null, localServerChannel0, channelBuffer0);
      jt600FrameDecoder0.decode((ChannelHandlerContext) null, localServerChannel0, byteBufferBackedChannelBuffer0);
      assertEquals(9, channelBuffer0.readerIndex());
      assertEquals(0, (int)localServerChannel0.getId());
  }
}
