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

package RootBeer.syr.pcpratts.rootbeer.runtime;

import java.util.Iterator;
import java.util.List;

public class Rootbeer implements IRootbeer {

  private IRootbeer m_Rootbeer;
  
  public Rootbeer(){
    RootbeerFactory factory = new RootbeerFactory();
    m_Rootbeer = factory.create();
  }

  public void runAll(List<Kernel> jobs) {
    m_Rootbeer.runAll(jobs);
  }

  public Iterator<Kernel> run(Iterator<Kernel> jobs) {
    return m_Rootbeer.run(jobs);
  }
}
