/*
 * This file was automatically generated by EvoSuite
 * Tue Dec 10 00:16:07 GMT 2019
 */

package org.traccar.events;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Collection;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.junit.runner.RunWith;
import org.traccar.Context;
import org.traccar.database.IdentityManager;
import org.traccar.events.IgnitionEventHandler;
import org.traccar.model.Device;
import org.traccar.model.Event;
import org.traccar.model.Position;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true, useJEE = true) 
public class IgnitionEventHandler_ESTest extends IgnitionEventHandler_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      IgnitionEventHandler ignitionEventHandler0 = new IgnitionEventHandler();
      IdentityManager identityManager0 = mock(IdentityManager.class, new ViolatedAssumptionAnswer());
      doReturn((Device) null, (Device) null, (Device) null, (Device) null, (Device) null).when(identityManager0).getDeviceById(anyLong());
      Context.init(identityManager0);
      Position position0 = new Position();
      MockDate mockDate0 = new MockDate((-3424L));
      mockDate0.toInstant();
      position0.setDeviceTime(mockDate0);
      MockDate.UTC(1642, 856, 76, 0, (-606), 0);
      ignitionEventHandler0.analyzePosition(position0);
      ignitionEventHandler0.analyzePosition(position0);
      position0.setServerTime(mockDate0);
      ignitionEventHandler0.analyzePosition(position0);
      ignitionEventHandler0.analyzePosition(position0);
      ignitionEventHandler0.analyzePosition(position0);
      ignitionEventHandler0.analyzePosition(position0);
      Position position1 = new Position();
      position1.set("geofenceExit", (-3424L));
      ignitionEventHandler0.analyzePosition(position1);
      ignitionEventHandler0.analyzePosition(position1);
      Position position2 = new Position();
      Collection<Event> collection0 = ignitionEventHandler0.analyzePosition(position2);
      assertNull(collection0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      IgnitionEventHandler ignitionEventHandler0 = new IgnitionEventHandler();
      Position position0 = new Position();
      // Undeclared exception!
      try { 
        ignitionEventHandler0.analyzePosition(position0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.traccar.events.IgnitionEventHandler", e);
      }
  }
}
