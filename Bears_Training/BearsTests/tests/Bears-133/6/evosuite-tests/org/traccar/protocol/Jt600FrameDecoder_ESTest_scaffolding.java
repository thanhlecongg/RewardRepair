/**
 * Scaffolding file used to store all the setups needed to run 
 * tests automatically generated by EvoSuite
 * Tue Dec 10 01:44:30 GMT 2019
 */

package org.traccar.protocol;

import org.evosuite.runtime.annotation.EvoSuiteClassExclude;
import org.junit.BeforeClass;
import org.junit.Before;
import org.junit.After;
import org.junit.AfterClass;
import org.evosuite.runtime.sandbox.Sandbox;
import org.evosuite.runtime.sandbox.Sandbox.SandboxMode;

@EvoSuiteClassExclude
public class Jt600FrameDecoder_ESTest_scaffolding {

  @org.junit.Rule 
  public org.evosuite.runtime.vnet.NonFunctionalRequirementRule nfr = new org.evosuite.runtime.vnet.NonFunctionalRequirementRule();

  private static final java.util.Properties defaultProperties = (java.util.Properties) java.lang.System.getProperties().clone(); 

  private org.evosuite.runtime.thread.ThreadStopper threadStopper =  new org.evosuite.runtime.thread.ThreadStopper (org.evosuite.runtime.thread.KillSwitchHandler.getInstance(), 3000);


  @BeforeClass 
  public static void initEvoSuiteFramework() { 
    org.evosuite.runtime.RuntimeSettings.className = "org.traccar.protocol.Jt600FrameDecoder"; 
    org.evosuite.runtime.GuiSupport.initialize(); 
    org.evosuite.runtime.RuntimeSettings.maxNumberOfThreads = 100; 
    org.evosuite.runtime.RuntimeSettings.maxNumberOfIterationsPerLoop = 10000; 
    org.evosuite.runtime.RuntimeSettings.mockSystemIn = true; 
    org.evosuite.runtime.RuntimeSettings.sandboxMode = org.evosuite.runtime.sandbox.Sandbox.SandboxMode.RECOMMENDED; 
    org.evosuite.runtime.sandbox.Sandbox.initializeSecurityManagerForSUT(); 
    org.evosuite.runtime.classhandling.JDKClassResetter.init();
    setSystemProperties();
    initializeClasses();
    org.evosuite.runtime.Runtime.getInstance().resetRuntime(); 
  } 

  @AfterClass 
  public static void clearEvoSuiteFramework(){ 
    Sandbox.resetDefaultSecurityManager(); 
    java.lang.System.setProperties((java.util.Properties) defaultProperties.clone()); 
  } 

  @Before 
  public void initTestCase(){ 
    threadStopper.storeCurrentThreads();
    threadStopper.startRecordingTime();
    org.evosuite.runtime.jvm.ShutdownHookHandler.getInstance().initHandler(); 
    org.evosuite.runtime.sandbox.Sandbox.goingToExecuteSUTCode(); 
    setSystemProperties(); 
    org.evosuite.runtime.GuiSupport.setHeadless(); 
    org.evosuite.runtime.Runtime.getInstance().resetRuntime(); 
    org.evosuite.runtime.agent.InstrumentingAgent.activate(); 
  } 

  @After 
  public void doneWithTestCase(){ 
    threadStopper.killAndJoinClientThreads();
    org.evosuite.runtime.jvm.ShutdownHookHandler.getInstance().safeExecuteAddedHooks(); 
    org.evosuite.runtime.classhandling.JDKClassResetter.reset(); 
    resetClasses(); 
    org.evosuite.runtime.sandbox.Sandbox.doneWithExecutingSUTCode(); 
    org.evosuite.runtime.agent.InstrumentingAgent.deactivate(); 
    org.evosuite.runtime.GuiSupport.restoreHeadlessMode(); 
  } 

  public static void setSystemProperties() {
 
    java.lang.System.setProperties((java.util.Properties) defaultProperties.clone()); 
    java.lang.System.setProperty("file.encoding", "UTF-8"); 
    java.lang.System.setProperty("java.awt.headless", "true"); 
    java.lang.System.setProperty("java.io.tmpdir", "/tmp"); 
    java.lang.System.setProperty("user.country", "US"); 
    java.lang.System.setProperty("user.dir", "/home/wasp/Desktop/research/Evosuite/tests/Bears-133/6"); 
    java.lang.System.setProperty("user.home", "/home/wasp"); 
    java.lang.System.setProperty("user.language", "en"); 
    java.lang.System.setProperty("user.name", "wasp"); 
    java.lang.System.setProperty("user.timezone", "America/Los_Angeles"); 
  }

  private static void initializeClasses() {
    org.evosuite.runtime.classhandling.ClassStateSupport.initializeClasses(Jt600FrameDecoder_ESTest_scaffolding.class.getClassLoader() ,
      "org.jboss.netty.buffer.ChannelBufferIndexFinder$9",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder$8",
      "org.jboss.netty.channel.socket.nio.AbstractNioChannel$WriteTask",
      "org.jboss.netty.channel.AdaptiveReceiveBufferSizePredictorFactory",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder$7",
      "org.jboss.netty.channel.local.LocalChannel",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder$6",
      "org.jboss.netty.channel.ChildChannelStateEvent",
      "org.jboss.netty.channel.DefaultChildChannelStateEvent",
      "org.jboss.netty.logging.JdkLoggerFactory",
      "org.jboss.netty.channel.socket.nio.AbstractNioSelector",
      "org.jboss.netty.channel.DefaultServerChannelConfig",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder$1",
      "org.jboss.netty.channel.CompleteChannelFuture",
      "com.cloudhopper.commons.util.Sequencer",
      "org.jboss.netty.channel.socket.nio.SocketSendBufferPool$UnpooledSendBuffer",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder$5",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder$4",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder$3",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder$2",
      "org.jboss.netty.channel.ChannelPipeline",
      "org.jboss.netty.channel.ChannelFutureListener",
      "org.jboss.netty.channel.socket.nio.SocketSendBufferPool$FileSendBuffer",
      "org.jboss.netty.buffer.DynamicChannelBuffer",
      "org.jboss.netty.channel.socket.DefaultSocketChannelConfig",
      "org.jboss.netty.channel.LifeCycleAwareChannelHandler",
      "org.jboss.netty.channel.ChannelHandlerLifeCycleException",
      "org.jboss.netty.channel.UpstreamMessageEvent",
      "org.jboss.netty.util.internal.DeadLockProofWorker",
      "org.jboss.netty.channel.group.DefaultChannelGroup$1",
      "org.jboss.netty.channel.ChannelException",
      "org.jboss.netty.channel.SimpleChannelUpstreamHandler$1",
      "org.jboss.netty.channel.socket.Worker",
      "org.jboss.netty.channel.SucceededChannelFuture",
      "org.jboss.netty.channel.group.ChannelGroup",
      "org.jboss.netty.channel.SimpleChannelUpstreamHandler",
      "org.jboss.netty.buffer.ByteBufferBackedChannelBuffer",
      "com.cloudhopper.commons.util.NamingThreadFactory",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder$10",
      "org.jboss.netty.channel.ChannelConfig",
      "org.jboss.netty.channel.socket.nio.NioSocketChannelConfig",
      "org.jboss.netty.channel.socket.nio.SocketSendBufferPool$SendBuffer",
      "org.jboss.netty.util.ThreadRenamingRunnable",
      "org.jboss.netty.util.internal.StringUtil",
      "org.jboss.netty.logging.AbstractInternalLogger",
      "org.jboss.netty.buffer.ReadOnlyChannelBuffer",
      "org.jboss.netty.channel.group.DefaultChannelGroupFuture",
      "org.jboss.netty.channel.ChannelEvent",
      "org.jboss.netty.buffer.LittleEndianHeapChannelBuffer",
      "org.jboss.netty.channel.group.DefaultChannelGroup",
      "org.jboss.netty.channel.socket.nio.NioSelector",
      "org.jboss.netty.channel.DownstreamMessageEvent",
      "org.jboss.netty.channel.socket.nio.SocketReceiveBufferAllocator",
      "org.jboss.netty.channel.local.LocalServerChannel",
      "com.zaxxer.hikari.util.UtilityElf",
      "org.jboss.netty.channel.SimpleChannelHandler",
      "org.jboss.netty.buffer.ChannelBuffers",
      "org.jboss.netty.channel.DefaultWriteCompletionEvent",
      "org.jboss.netty.channel.ServerChannel",
      "org.jboss.netty.channel.DownstreamChannelStateEvent$1",
      "org.traccar.helper.StringFinder",
      "org.jboss.netty.channel.ChannelUpstreamHandler",
      "org.jboss.netty.channel.ChannelHandler",
      "org.jboss.netty.logging.InternalLogLevel",
      "org.jboss.netty.channel.socket.nio.NioWorker",
      "org.jboss.netty.channel.local.DefaultLocalServerChannelFactory",
      "org.jboss.netty.channel.SimpleChannelDownstreamHandler",
      "org.jboss.netty.channel.Channel",
      "org.jboss.netty.channel.socket.nio.SocketSendBufferPool$Preallocation",
      "org.jboss.netty.channel.local.LocalClientChannelSink",
      "org.jboss.netty.channel.socket.nio.AbstractNioChannel$WriteRequestQueue",
      "org.jboss.netty.channel.AbstractChannel",
      "org.jboss.netty.channel.UpstreamChannelStateEvent",
      "org.jboss.netty.logging.JdkLogger",
      "org.jboss.netty.channel.WriteCompletionEvent",
      "org.jboss.netty.channel.FailedChannelFuture",
      "org.jboss.netty.channel.AbstractChannelSink",
      "org.jboss.netty.buffer.HeapChannelBuffer",
      "org.jboss.netty.buffer.SlicedChannelBuffer",
      "org.jboss.netty.buffer.TruncatedChannelBuffer",
      "org.jboss.netty.channel.socket.nio.NioSocketChannel",
      "org.jboss.netty.channel.local.DefaultLocalServerChannel",
      "org.jboss.netty.channel.ChannelFutureProgressListener",
      "org.jboss.netty.channel.local.LocalServerChannelFactory",
      "org.jboss.netty.buffer.ChannelBufferFactory",
      "org.jboss.netty.util.ThreadNameDeterminer",
      "org.jboss.netty.util.ThreadNameDeterminer$2",
      "org.jboss.netty.util.ThreadNameDeterminer$1",
      "org.jboss.netty.buffer.ChannelBuffer",
      "org.jboss.netty.channel.group.ChannelGroupFuture",
      "org.jboss.netty.channel.group.DefaultChannelGroupFuture$1",
      "org.jboss.netty.buffer.AbstractChannelBuffer",
      "org.jboss.netty.buffer.DuplicatedChannelBuffer",
      "org.jboss.netty.channel.socket.nio.NioChannelConfig",
      "org.jboss.netty.buffer.BigEndianHeapChannelBuffer",
      "org.jboss.netty.channel.socket.nio.SocketSendBufferPool$PreallocationRef",
      "org.jboss.netty.channel.ChannelState",
      "org.jboss.netty.logging.InternalLoggerFactory",
      "org.jboss.netty.channel.local.LocalAddress",
      "org.jboss.netty.channel.Channels",
      "org.jboss.netty.channel.socket.SocketChannelConfig",
      "org.jboss.netty.channel.ExceptionEvent",
      "org.jboss.netty.util.internal.SystemPropertyUtil",
      "org.jboss.netty.channel.ChannelFutureListener$2",
      "org.jboss.netty.channel.DefaultChannelPipeline",
      "org.jboss.netty.channel.AbstractChannel$ChannelCloseFuture",
      "org.jboss.netty.channel.local.LocalServerChannelSink$1",
      "org.jboss.netty.channel.local.DefaultLocalChannel$1",
      "org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext",
      "org.jboss.netty.channel.ChannelPipelineFactory",
      "org.jboss.netty.util.internal.ThreadLocalBoolean",
      "org.jboss.netty.channel.ChannelFutureListener$1",
      "com.zaxxer.hikari.util.UtilityElf$DefaultThreadFactory",
      "org.traccar.protocol.Jt600FrameDecoder",
      "org.jboss.netty.channel.DefaultExceptionEvent",
      "org.jboss.netty.buffer.AbstractChannelBufferFactory",
      "org.jboss.netty.channel.socket.nio.AbstractNioChannel",
      "org.jboss.netty.channel.socket.SocketChannel",
      "org.jboss.netty.logging.InternalLogger",
      "org.jboss.netty.channel.local.DefaultLocalClientChannelFactory",
      "org.jboss.netty.util.ExternalResourceReleasable",
      "org.jboss.netty.channel.UpstreamChannelStateEvent$1",
      "org.jboss.netty.buffer.CompositeChannelBuffer",
      "org.jboss.netty.util.internal.DeadLockProofWorker$1",
      "org.jboss.netty.buffer.HeapChannelBufferFactory",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder",
      "org.jboss.netty.channel.ChannelHandlerContext",
      "org.jboss.netty.channel.DefaultChannelPipeline$DiscardingChannelSink",
      "org.jboss.netty.channel.ReceiveBufferSizePredictorFactory",
      "org.jboss.netty.buffer.WrappedChannelBuffer",
      "org.jboss.netty.channel.ChannelFuture",
      "org.jboss.netty.channel.socket.nio.SocketSendBufferPool",
      "org.jboss.netty.channel.AbstractServerChannel",
      "org.jboss.netty.channel.local.LocalServerChannelSink",
      "org.jboss.netty.channel.ReceiveBufferSizePredictor",
      "org.jboss.netty.buffer.DirectChannelBufferFactory",
      "org.jboss.netty.channel.FileRegion",
      "org.jboss.netty.channel.socket.nio.SocketSendBufferPool$PooledSendBuffer",
      "org.jboss.netty.channel.socket.nio.SocketSendBufferPool$EmptySendBuffer",
      "org.jboss.netty.channel.ChannelSink",
      "org.jboss.netty.channel.ServerChannelFactory",
      "org.jboss.netty.channel.local.DefaultLocalChannel",
      "org.jboss.netty.channel.socket.nio.DefaultNioSocketChannelConfig",
      "org.jboss.netty.channel.socket.nio.AbstractNioWorker",
      "org.jboss.netty.channel.ChannelPipelineException",
      "org.jboss.netty.buffer.EmptyChannelBuffer",
      "org.jboss.netty.channel.DefaultChannelConfig",
      "org.jboss.netty.channel.socket.nio.SocketSendBufferPool$GatheringSendBuffer",
      "org.jboss.netty.channel.ChannelDownstreamHandler",
      "org.jboss.netty.channel.MessageEvent",
      "org.jboss.netty.channel.DownstreamChannelStateEvent",
      "org.jboss.netty.handler.codec.frame.FrameDecoder",
      "org.jboss.netty.channel.ChannelStateEvent",
      "org.jboss.netty.channel.local.LocalClientChannelFactory",
      "org.jboss.netty.channel.group.ChannelGroupFutureListener",
      "org.jboss.netty.channel.socket.nio.SelectorUtil",
      "org.jboss.netty.channel.DefaultChannelFuture",
      "org.jboss.netty.channel.ChannelFactory"
    );
  } 

  private static void resetClasses() {
    org.evosuite.runtime.classhandling.ClassResetter.getInstance().setClassLoader(Jt600FrameDecoder_ESTest_scaffolding.class.getClassLoader()); 

    org.evosuite.runtime.classhandling.ClassStateSupport.resetClasses(
      "org.jboss.netty.logging.JdkLoggerFactory",
      "org.jboss.netty.logging.InternalLoggerFactory",
      "org.jboss.netty.logging.AbstractInternalLogger",
      "org.jboss.netty.logging.JdkLogger",
      "org.jboss.netty.channel.SimpleChannelUpstreamHandler",
      "org.jboss.netty.handler.codec.frame.FrameDecoder",
      "org.traccar.protocol.Jt600FrameDecoder",
      "org.jboss.netty.channel.DefaultChannelPipeline$DiscardingChannelSink",
      "org.jboss.netty.channel.DefaultChannelPipeline",
      "org.jboss.netty.buffer.AbstractChannelBuffer",
      "org.jboss.netty.buffer.DynamicChannelBuffer",
      "org.jboss.netty.buffer.AbstractChannelBufferFactory",
      "org.jboss.netty.buffer.HeapChannelBufferFactory",
      "org.jboss.netty.buffer.HeapChannelBuffer",
      "org.jboss.netty.buffer.BigEndianHeapChannelBuffer",
      "org.jboss.netty.buffer.EmptyChannelBuffer",
      "org.jboss.netty.buffer.ChannelBuffers",
      "org.jboss.netty.buffer.TruncatedChannelBuffer",
      "org.jboss.netty.buffer.DirectChannelBufferFactory",
      "org.jboss.netty.channel.local.DefaultLocalClientChannelFactory",
      "org.jboss.netty.channel.AbstractChannelSink",
      "org.jboss.netty.channel.local.LocalClientChannelSink",
      "org.jboss.netty.channel.AbstractChannel",
      "org.jboss.netty.channel.local.DefaultLocalChannel",
      "org.jboss.netty.channel.CompleteChannelFuture",
      "org.jboss.netty.channel.SucceededChannelFuture",
      "org.jboss.netty.channel.DefaultChannelFuture",
      "org.jboss.netty.channel.AbstractChannel$ChannelCloseFuture",
      "org.jboss.netty.util.internal.ThreadLocalBoolean",
      "org.jboss.netty.channel.DefaultChannelConfig",
      "org.jboss.netty.channel.local.DefaultLocalChannel$1",
      "org.jboss.netty.channel.Channels",
      "org.jboss.netty.channel.UpstreamChannelStateEvent",
      "org.jboss.netty.channel.UpstreamChannelStateEvent$1",
      "org.jboss.netty.channel.local.DefaultLocalServerChannelFactory",
      "org.jboss.netty.channel.group.DefaultChannelGroup",
      "org.jboss.netty.channel.group.DefaultChannelGroup$1",
      "org.jboss.netty.channel.local.LocalServerChannelSink",
      "org.jboss.netty.channel.AbstractServerChannel",
      "org.jboss.netty.channel.local.DefaultLocalServerChannel",
      "org.jboss.netty.channel.DefaultServerChannelConfig",
      "org.jboss.netty.buffer.LittleEndianHeapChannelBuffer",
      "org.jboss.netty.buffer.ByteBufferBackedChannelBuffer",
      "org.jboss.netty.channel.group.DefaultChannelGroupFuture",
      "org.jboss.netty.channel.group.DefaultChannelGroupFuture$1",
      "org.jboss.netty.buffer.DuplicatedChannelBuffer",
      "org.jboss.netty.channel.DefaultChildChannelStateEvent",
      "org.jboss.netty.buffer.ReadOnlyChannelBuffer",
      "org.jboss.netty.buffer.SlicedChannelBuffer",
      "com.cloudhopper.commons.util.NamingThreadFactory",
      "com.cloudhopper.commons.util.Sequencer",
      "org.jboss.netty.channel.socket.nio.AbstractNioSelector",
      "org.jboss.netty.channel.socket.nio.AbstractNioWorker",
      "org.jboss.netty.channel.socket.nio.NioWorker",
      "org.jboss.netty.util.internal.SystemPropertyUtil",
      "org.jboss.netty.channel.socket.nio.SelectorUtil",
      "org.jboss.netty.util.ThreadNameDeterminer$1",
      "org.jboss.netty.util.ThreadNameDeterminer$2",
      "org.jboss.netty.util.ThreadNameDeterminer",
      "org.jboss.netty.util.ThreadRenamingRunnable",
      "org.jboss.netty.util.internal.DeadLockProofWorker",
      "org.jboss.netty.util.internal.DeadLockProofWorker$1",
      "org.jboss.netty.channel.socket.nio.SocketSendBufferPool$EmptySendBuffer",
      "org.jboss.netty.channel.socket.nio.SocketSendBufferPool",
      "org.jboss.netty.channel.socket.nio.SocketSendBufferPool$Preallocation",
      "org.jboss.netty.channel.socket.nio.SocketReceiveBufferAllocator",
      "org.jboss.netty.channel.socket.nio.AbstractNioChannel",
      "org.jboss.netty.channel.socket.nio.NioSocketChannel",
      "org.jboss.netty.channel.socket.nio.AbstractNioChannel$WriteTask",
      "org.jboss.netty.channel.socket.nio.AbstractNioChannel$WriteRequestQueue",
      "org.jboss.netty.channel.socket.DefaultSocketChannelConfig",
      "org.jboss.netty.channel.AdaptiveReceiveBufferSizePredictorFactory",
      "org.jboss.netty.channel.socket.nio.DefaultNioSocketChannelConfig",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder$1",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder$2",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder$3",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder$4",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder$5",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder$6",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder$7",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder$8",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder$9",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder$10",
      "org.jboss.netty.buffer.ChannelBufferIndexFinder",
      "org.jboss.netty.buffer.CompositeChannelBuffer",
      "org.jboss.netty.channel.FailedChannelFuture",
      "org.jboss.netty.channel.DownstreamChannelStateEvent",
      "org.jboss.netty.channel.local.LocalServerChannelSink$1",
      "org.jboss.netty.channel.SimpleChannelDownstreamHandler",
      "org.jboss.netty.channel.DefaultWriteCompletionEvent",
      "org.jboss.netty.channel.ChannelException",
      "org.jboss.netty.channel.UpstreamMessageEvent",
      "com.zaxxer.hikari.util.UtilityElf$DefaultThreadFactory",
      "org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext",
      "org.jboss.netty.channel.DefaultExceptionEvent",
      "org.jboss.netty.channel.SimpleChannelUpstreamHandler$1",
      "org.traccar.helper.StringFinder",
      "org.jboss.netty.channel.SimpleChannelHandler",
      "org.jboss.netty.channel.DownstreamChannelStateEvent$1",
      "org.jboss.netty.channel.DownstreamMessageEvent",
      "org.jboss.netty.util.internal.StringUtil",
      "org.jboss.netty.channel.ChannelState"
    );
  }
}
