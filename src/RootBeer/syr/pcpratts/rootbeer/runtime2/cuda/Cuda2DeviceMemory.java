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

package RootBeer.syr.pcpratts.rootbeer.runtime2.cuda;

import RootBeer.syr.pcpratts.rootbeer.runtime.memory.DeviceMemory;

public class Cuda2DeviceMemory implements DeviceMemory {

  public Cuda2DeviceMemory(long cpu_addr, long gpu_addr){
    setup(cpu_addr, gpu_addr);
  }
  
  public native void read(byte[] curr_block, long index, int size);
  public native void write(byte[] curr_block, long index, int size);
  private native void setup(long cpu_addr, long gpu_addr);
}
