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

package RootBeer.syr.pcpratts.jpp;

import RootBeer.syr.pcpratts.jpp.cfile.CFile;
import RootBeer.syr.pcpratts.jpp.cfile.CFileItem;
import RootBeer.syr.pcpratts.jpp.cfile.internal.CFileFactory;
import RootBeer.syr.pcpratts.jpp.parser.ParserFactory;
import RootBeer.syr.pcpratts.jpp.parser.PutbackIterator;
import RootBeer.syr.pcpratts.jpp.parser.preprocessor.IncludePath;
import RootBeer.syr.pcpratts.jpp.typetable.TypeTable;
import java.io.FileNotFoundException;

public class Jpp {

  private IncludePath m_IncludePath;

  public Jpp(boolean is_cpp) {
    m_IncludePath = new IncludePath();
    JppSettings.v().setCpp(is_cpp);
  }

  public void addIncludeFolder(String path){
    m_IncludePath.addFolder(path);
  }

  public void translateFile(String filename) throws FileNotFoundException {

    //Preprocessor preproc = new Preprocessor(m_IncludePath);
    //Reader reader = preproc.process(filename);

    ParserFactory pfactory = new ParserFactory();
    PutbackIterator<CFileItem> iter = pfactory.createCppFromFilename(filename);

    CFileFactory cfactory = new CFileFactory();
    CFile cfile = cfactory.create(iter);
    
    TypeTable type_table = new TypeTable(cfile);

    System.out.println(cfile.toString());
  }
}
