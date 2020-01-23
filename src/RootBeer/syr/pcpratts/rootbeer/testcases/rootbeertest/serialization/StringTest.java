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

package RootBeer.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization;

import RootBeer.syr.pcpratts.rootbeer.runtime.Kernel;
import RootBeer.syr.pcpratts.rootbeer.test.TestSerialization;
import java.util.ArrayList;
import java.util.List;

public class StringTest implements TestSerialization {

  @Override
  public List<Kernel> create() {
    List<Kernel> jobs = new ArrayList<Kernel>();
    for(int i = 0; i < 100; ++i){
      StringRunOnGpu curr = new StringRunOnGpu(i);
      jobs.add(curr);
    }
    return jobs;
  }

  @Override
  public boolean compare(Kernel lhs, Kernel rhs) {
    StringRunOnGpu blhs = (StringRunOnGpu) lhs;
    StringRunOnGpu brhs = (StringRunOnGpu) rhs;
    return blhs.compare(brhs);
  }
}
