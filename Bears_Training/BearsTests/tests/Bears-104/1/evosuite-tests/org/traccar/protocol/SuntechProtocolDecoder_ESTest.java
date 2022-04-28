/*
 * This file was automatically generated by EvoSuite
 * Tue Dec 10 00:39:51 GMT 2019
 */

package org.traccar.protocol;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;
import org.traccar.protocol.SuntechProtocol;
import org.traccar.protocol.SuntechProtocolDecoder;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true, useJEE = true) 
public class SuntechProtocolDecoder_ESTest extends SuntechProtocolDecoder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      SuntechProtocol suntechProtocol0 = new SuntechProtocol();
      SuntechProtocolDecoder suntechProtocolDecoder0 = null;
      try {
        suntechProtocolDecoder0 = new SuntechProtocolDecoder(suntechProtocol0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.traccar.protocol.SuntechProtocolDecoder", e);
      }
  }
}
