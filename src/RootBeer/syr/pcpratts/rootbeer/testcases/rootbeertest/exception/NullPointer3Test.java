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

package RootBeer.syr.pcpratts.rootbeer.testcases.rootbeertest.exception;

import RootBeer.syr.pcpratts.rootbeer.runtime.Kernel;
import RootBeer.syr.pcpratts.rootbeer.test.TestException;
import java.util.ArrayList;
import java.util.List;

public class NullPointer3Test implements TestException {

  @Override
  public List<Kernel> create() {
    List<Kernel> jobs = new ArrayList<Kernel>();
    for(int i = 0; i < 4096; ++i){
      NullPointer3RunOnGpu curr = new NullPointer3RunOnGpu();
      jobs.add(curr);
    }
    return jobs;
  }

  @Override
  public boolean catchException(Throwable thrwbl) {
    if(thrwbl instanceof NullPointerException){
      return true; 
    }
    return false;
  }
  
}
