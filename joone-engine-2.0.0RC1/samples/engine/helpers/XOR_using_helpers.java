/*
 * XOR_using_helpers.java
 *
 * Created on January 17, 2006, 5:21 PM
 *
 * Copyright @2005 by Paolo Marrone and the Joone team
 * Licensed under the Lesser General Public License (LGPL);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.gnu.org/
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.joone.samples.engine.helpers;

import org.joone.helpers.factory.JooneTools;
import org.joone.net.NeuralNet;

/**
 * Example to demonstrate the use of the helpers classes
 * @author P.Marrone
 */
public class XOR_using_helpers {
    
    // XOR input
    private static double[][]	inputArray = new double[][] {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    
    // XOR desired output
    private static double[][]	desiredArray = new double[][] {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };
    
    private static boolean singleThreadMode = true;
    /**
     * Creates a new instance of XOR_using_helpers
     */
    public XOR_using_helpers() {
    }
    
    public static void main(String[] args) {
        try {
        // Create the network: 3 layers with a logistic output layer
        NeuralNet nnet = JooneTools.create_standard(new int[]{ 2, 2, 1 }, JooneTools.LOGISTIC);
//            NeuralNet nnet = JooneTools.load("org/joone/samples/engine/helpers/rxor.snet");
            nnet.getMonitor().setSingleThreadMode(singleThreadMode);
            // Train the network for 5000 epochs, or until rmse reaches 0.01.
            // Outputs the results every 200 epochs on the stardard output
            double rmse = JooneTools.train(nnet, inputArray, desiredArray,
                    5000, 0.01,
                    200, System.out, false);
            
            // Waits in order to avoid the interlacing of the rows displayed
            try { Thread.sleep(50); } catch (InterruptedException doNothing) { }
            
            // Interrogate the network and prints the results
            System.out.println("Last RMSE = "+rmse);
            System.out.println("\nResults:");
            System.out.println("|Inp 1\t|Inp 2\t|Output");
            for (int i=0; i < 4; ++i) {
                double[] output = JooneTools.interrogate(nnet, inputArray[i]);
                System.out.print("| "+inputArray[i][0]+"\t| "+inputArray[i][1]+"\t| ");
                System.out.println(output[0]);
            }
            
            // Test the network and prints the rmse
            double testRMSE = JooneTools.test(nnet, inputArray, desiredArray);
            System.out.println("\nTest error = "+testRMSE);
        } catch (Exception exc) { exc.printStackTrace(); }
    }
}
