/*
 * This file is part of Rootbeer.
 * 
 * Rootbeer is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Rootbeer is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Rootbeer.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

package RootBeer.syr.pcpratts.rootbeer.testcases.rootbeertest.gpurequired;

import RootBeer.syr.pcpratts.rootbeer.runtime.Kernel;

import java.util.ArrayList;
import java.util.List;

public class SimpleSynchronizedRunOnGpu implements Kernel {

  private SimpleSynchronizedObject m_syncObj;
  private List<Integer> m_olds;
  
  public SimpleSynchronizedRunOnGpu(SimpleSynchronizedObject sync_obj){
    m_syncObj = sync_obj;
    m_olds = new ArrayList<Integer>();
  }
  
  public void gpuMethod() {
    m_syncObj.inc();
  }
  
  public SimpleSynchronizedObject getSyncObj(){
    return m_syncObj;
  }
}
