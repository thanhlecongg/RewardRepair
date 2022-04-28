/**
 * Scaffolding file used to store all the setups needed to run 
 * tests automatically generated by EvoSuite
 * Tue Dec 10 19:58:33 GMT 2019
 */

package com.alibaba.dubbo.rpc.cluster.router.condition;

import org.evosuite.runtime.annotation.EvoSuiteClassExclude;
import org.junit.BeforeClass;
import org.junit.Before;
import org.junit.After;
import org.junit.AfterClass;
import org.evosuite.runtime.sandbox.Sandbox;
import org.evosuite.runtime.sandbox.Sandbox.SandboxMode;

@EvoSuiteClassExclude
public class ConditionRouter_ESTest_scaffolding {

  @org.junit.Rule 
  public org.evosuite.runtime.vnet.NonFunctionalRequirementRule nfr = new org.evosuite.runtime.vnet.NonFunctionalRequirementRule();

  private static final java.util.Properties defaultProperties = (java.util.Properties) java.lang.System.getProperties().clone(); 

  private org.evosuite.runtime.thread.ThreadStopper threadStopper =  new org.evosuite.runtime.thread.ThreadStopper (org.evosuite.runtime.thread.KillSwitchHandler.getInstance(), 3000);


  @BeforeClass 
  public static void initEvoSuiteFramework() { 
    org.evosuite.runtime.RuntimeSettings.className = "com.alibaba.dubbo.rpc.cluster.router.condition.ConditionRouter"; 
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
    java.lang.System.setProperty("user.dir", "/home/wasp/Desktop/research/Evosuite/tests/Bears-149/5"); 
    java.lang.System.setProperty("user.home", "/home/wasp"); 
    java.lang.System.setProperty("user.language", "en"); 
    java.lang.System.setProperty("user.name", "wasp"); 
    java.lang.System.setProperty("user.timezone", "America/Los_Angeles"); 
  }

  private static void initializeClasses() {
    org.evosuite.runtime.classhandling.ClassStateSupport.initializeClasses(ConditionRouter_ESTest_scaffolding.class.getClassLoader() ,
      "com.alibaba.dubbo.common.compiler.support.JavassistCompiler",
      "com.alibaba.dubbo.common.logger.support.FailsafeLogger",
      "com.alibaba.dubbo.common.utils.ClassHelper",
      "com.alibaba.dubbo.common.compiler.support.AdaptiveCompiler",
      "com.alibaba.dubbo.common.utils.StringUtils",
      "com.alibaba.dubbo.common.extension.factory.AdaptiveExtensionFactory",
      "com.alibaba.dubbo.rpc.Protocol",
      "com.alibaba.dubbo.common.Version",
      "com.alibaba.dubbo.common.utils.LRUCache",
      "com.alibaba.dubbo.common.extension.Activate",
      "com.alibaba.dubbo.rpc.cluster.router.condition.ConditionRouter$MatchPair",
      "com.alibaba.dubbo.rpc.listener.ListenerInvokerWrapper",
      "com.alibaba.dubbo.rpc.proxy.wrapper.StubProxyFactoryWrapper",
      "com.alibaba.dubbo.rpc.cluster.router.condition.ConditionRouter",
      "com.alibaba.dubbo.common.compiler.support.AbstractCompiler",
      "com.alibaba.dubbo.common.Constants",
      "com.alibaba.dubbo.rpc.RpcException",
      "com.alibaba.dubbo.rpc.Exporter",
      "com.alibaba.dubbo.rpc.ProxyFactory",
      "com.alibaba.dubbo.rpc.proxy.AbstractProxyFactory",
      "com.alibaba.dubbo.common.logger.LoggerFactory",
      "com.alibaba.dubbo.common.URL",
      "com.alibaba.dubbo.rpc.RpcInvocation",
      "com.alibaba.dubbo.common.extension.ExtensionLoader",
      "com.alibaba.dubbo.config.spring.extension.SpringExtensionFactory",
      "com.alibaba.dubbo.common.extension.factory.SpiExtensionFactory",
      "com.alibaba.dubbo.common.utils.ConcurrentHashSet",
      "com.alibaba.dubbo.common.logger.Logger",
      "com.alibaba.dubbo.rpc.Invoker",
      "com.alibaba.dubbo.rpc.Result",
      "com.alibaba.dubbo.common.logger.slf4j.Slf4jLogger",
      "com.alibaba.dubbo.rpc.InvokerListener",
      "com.alibaba.dubbo.common.extension.SPI",
      "com.alibaba.dubbo.common.logger.Level",
      "com.alibaba.dubbo.common.utils.CollectionUtils",
      "com.alibaba.dubbo.common.io.UnsafeStringWriter",
      "com.alibaba.dubbo.common.utils.Holder",
      "com.alibaba.dubbo.rpc.Invocation",
      "com.alibaba.dubbo.rpc.support.MockInvoker",
      "com.alibaba.dubbo.common.compiler.support.JdkCompiler",
      "com.alibaba.dubbo.common.utils.CollectionUtils$1",
      "com.alibaba.dubbo.common.extension.ExtensionFactory",
      "com.alibaba.dubbo.rpc.cluster.Router",
      "com.alibaba.dubbo.common.compiler.Compiler",
      "com.alibaba.dubbo.common.utils.NetUtils",
      "com.alibaba.dubbo.rpc.proxy.javassist.JavassistProxyFactory",
      "com.alibaba.dubbo.common.logger.slf4j.Slf4jLoggerAdapter",
      "com.alibaba.dubbo.common.logger.LoggerAdapter",
      "com.alibaba.dubbo.rpc.cluster.router.MockInvokersSelector",
      "com.alibaba.dubbo.rpc.protocol.InvokerWrapper",
      "com.alibaba.dubbo.common.Node",
      "com.alibaba.dubbo.common.extension.Adaptive",
      "com.alibaba.dubbo.rpc.proxy.jdk.JdkProxyFactory"
    );
  } 

  private static void resetClasses() {
    org.evosuite.runtime.classhandling.ClassResetter.getInstance().setClassLoader(ConditionRouter_ESTest_scaffolding.class.getClassLoader()); 

    org.evosuite.runtime.classhandling.ClassStateSupport.resetClasses(
      "com.alibaba.dubbo.common.logger.slf4j.Slf4jLoggerAdapter",
      "com.alibaba.dubbo.common.logger.slf4j.Slf4jLogger",
      "com.alibaba.dubbo.common.logger.LoggerFactory",
      "com.alibaba.dubbo.common.logger.support.FailsafeLogger",
      "com.alibaba.dubbo.rpc.cluster.router.condition.ConditionRouter",
      "com.alibaba.dubbo.rpc.cluster.router.condition.ConditionRouter$MatchPair",
      "com.alibaba.dubbo.common.utils.LRUCache",
      "com.alibaba.dubbo.common.utils.NetUtils",
      "com.alibaba.dubbo.common.URL",
      "com.alibaba.dubbo.common.utils.CollectionUtils$1",
      "com.alibaba.dubbo.common.utils.CollectionUtils",
      "com.alibaba.dubbo.rpc.cluster.router.MockInvokersSelector",
      "com.alibaba.dubbo.common.utils.StringUtils",
      "com.alibaba.dubbo.rpc.protocol.InvokerWrapper",
      "com.alibaba.dubbo.rpc.RpcInvocation",
      "com.alibaba.dubbo.common.extension.ExtensionLoader",
      "com.alibaba.dubbo.common.utils.Holder",
      "com.alibaba.dubbo.common.utils.ConcurrentHashSet",
      "com.alibaba.dubbo.config.spring.extension.SpringExtensionFactory",
      "com.alibaba.dubbo.common.extension.factory.AdaptiveExtensionFactory",
      "com.alibaba.dubbo.common.extension.factory.SpiExtensionFactory",
      "com.alibaba.dubbo.rpc.proxy.wrapper.StubProxyFactoryWrapper",
      "com.alibaba.dubbo.rpc.proxy.AbstractProxyFactory",
      "com.alibaba.dubbo.rpc.proxy.jdk.JdkProxyFactory",
      "com.alibaba.dubbo.rpc.proxy.javassist.JavassistProxyFactory",
      "com.alibaba.dubbo.common.compiler.support.AdaptiveCompiler",
      "com.alibaba.dubbo.common.compiler.support.AbstractCompiler",
      "com.alibaba.dubbo.common.compiler.support.JdkCompiler",
      "com.alibaba.dubbo.rpc.support.MockInvoker",
      "com.alibaba.dubbo.rpc.listener.ListenerInvokerWrapper",
      "com.alibaba.dubbo.common.Constants",
      "com.alibaba.dubbo.common.utils.ClassHelper",
      "com.alibaba.dubbo.common.Version"
    );
  }
}
