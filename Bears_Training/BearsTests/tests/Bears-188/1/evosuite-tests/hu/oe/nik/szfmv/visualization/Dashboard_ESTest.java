/*
 * This file was automatically generated by EvoSuite
 * Tue Dec 10 18:52:07 GMT 2019
 */

package hu.oe.nik.szfmv.visualization;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import hu.oe.nik.szfmv.automatedcar.bus.packets.input.ReadOnlyInputPacket;
import hu.oe.nik.szfmv.visualization.Dashboard;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true, useJEE = true) 
public class Dashboard_ESTest extends Dashboard_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Dashboard dashboard0 = new Dashboard();
      // Undeclared exception!
      try { 
        dashboard0.paintComponent((Graphics) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("javax.swing.JComponent", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Dashboard dashboard0 = new Dashboard();
      ReadOnlyInputPacket readOnlyInputPacket0 = mock(ReadOnlyInputPacket.class, new ViolatedAssumptionAnswer());
      doReturn(417).when(readOnlyInputPacket0).getBreakPedalPosition();
      doReturn(0).when(readOnlyInputPacket0).getGasPedalPosition();
      dashboard0.updateDisplayedValues(readOnlyInputPacket0);
      BufferedImage bufferedImage0 = new BufferedImage(255, 417, 2);
      Graphics graphics0 = bufferedImage0.getGraphics();
      dashboard0.paintComponent(graphics0);
      assertFalse(dashboard0.isFocusCycleRoot());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Dashboard dashboard0 = new Dashboard();
      // Undeclared exception!
      try { 
        dashboard0.updateDisplayedValues((ReadOnlyInputPacket) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("hu.oe.nik.szfmv.visualization.Dashboard", e);
      }
  }
}
