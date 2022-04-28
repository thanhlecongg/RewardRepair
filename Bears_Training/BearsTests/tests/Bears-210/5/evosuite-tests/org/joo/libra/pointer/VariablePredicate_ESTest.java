/*
 * This file was automatically generated by EvoSuite
 * Tue Dec 10 14:33:48 GMT 2019
 */

package org.joo.libra.pointer;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.joo.libra.PredicateContext;
import org.joo.libra.common.HasValue;
import org.joo.libra.common.SimpleHasValue;
import org.joo.libra.pointer.VariablePredicate;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true, useJEE = true) 
public class VariablePredicate_ESTest extends VariablePredicate_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Double double0 = new Double((-555.22));
      SimpleHasValue<Object> simpleHasValue0 = new SimpleHasValue<Object>(double0);
      VariablePredicate variablePredicate0 = new VariablePredicate(simpleHasValue0);
      PredicateContext predicateContext0 = new PredicateContext(double0);
      boolean boolean0 = variablePredicate0.satisfiedBy(predicateContext0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      VariablePredicate variablePredicate0 = new VariablePredicate((HasValue<Object>) null);
      Object object0 = new Object();
      PredicateContext predicateContext0 = new PredicateContext(object0);
      // Undeclared exception!
      try { 
        variablePredicate0.satisfiedBy(predicateContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.joo.libra.pointer.VariablePredicate", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Object object0 = new Object();
      SimpleHasValue<Object> simpleHasValue0 = new SimpleHasValue<Object>(object0);
      VariablePredicate variablePredicate0 = new VariablePredicate(simpleHasValue0);
      PredicateContext predicateContext0 = new PredicateContext(variablePredicate0);
      boolean boolean0 = variablePredicate0.satisfiedBy(predicateContext0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Short short0 = new Short((short)3133);
      SimpleHasValue<Object> simpleHasValue0 = new SimpleHasValue<Object>(short0);
      VariablePredicate variablePredicate0 = new VariablePredicate(simpleHasValue0);
      PredicateContext predicateContext0 = new PredicateContext(short0);
      boolean boolean0 = variablePredicate0.satisfiedBy(predicateContext0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      SimpleHasValue<Object> simpleHasValue0 = new SimpleHasValue<Object>("9");
      PredicateContext predicateContext0 = new PredicateContext("9");
      VariablePredicate variablePredicate0 = new VariablePredicate(simpleHasValue0);
      boolean boolean0 = variablePredicate0.satisfiedBy(predicateContext0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      SimpleHasValue<Object> simpleHasValue0 = new SimpleHasValue<Object>("");
      VariablePredicate variablePredicate0 = new VariablePredicate(simpleHasValue0);
      boolean boolean0 = variablePredicate0.satisfiedBy((PredicateContext) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      SimpleHasValue<Object> simpleHasValue0 = new SimpleHasValue<Object>((Object) null);
      VariablePredicate variablePredicate0 = new VariablePredicate(simpleHasValue0);
      boolean boolean0 = variablePredicate0.satisfiedBy((PredicateContext) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Float float0 = new Float(0.0F);
      SimpleHasValue<Object> simpleHasValue0 = new SimpleHasValue<Object>(float0);
      VariablePredicate variablePredicate0 = new VariablePredicate(simpleHasValue0);
      PredicateContext predicateContext0 = new PredicateContext(variablePredicate0);
      boolean boolean0 = variablePredicate0.satisfiedBy(predicateContext0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Float float0 = new Float(0.0F);
      SimpleHasValue<Object> simpleHasValue0 = new SimpleHasValue<Object>(float0);
      VariablePredicate variablePredicate0 = new VariablePredicate(simpleHasValue0);
      String string0 = variablePredicate0.toString();
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      Boolean boolean0 = Boolean.TRUE;
      SimpleHasValue<Object> simpleHasValue0 = new SimpleHasValue<Object>(boolean0);
      VariablePredicate variablePredicate0 = new VariablePredicate(simpleHasValue0);
      PredicateContext predicateContext0 = new PredicateContext(boolean0);
      boolean boolean1 = variablePredicate0.satisfiedBy(predicateContext0);
      assertTrue(boolean1);
  }
}
