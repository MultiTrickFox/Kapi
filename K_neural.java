import java.util.ArrayList;


// 2DO: compile as .jar

class K_neural {
  
    K_base base;

    // constructor
    
    K_neural() {
      
      this.base = new K_base();
      
    }
    
    // initializers
    
    ArrayList<ArrayList<Double[][]>> make_gru(int in_size, ArrayList<Integer> hidden_sizes, int out_size) {
      
        ArrayList<ArrayList<Double[][]>> gru = new ArrayList();
        
        int hm_layers = hidden_sizes.size()+1;
        
        for (int l = 0; l < hm_layers; l++) {
          
            int i,o;
            ArrayList<Double[][]> layer = new ArrayList();
            
            if      (l == 0)           { i = in_size ; o = hidden_sizes.get(0); }
            else if (l == hm_layers-1) { i = hidden_sizes.get(l-1) ; o = hidden_sizes.get(l); }
            else                       { i = hidden_sizes.get(l) ; o = out_size; }
                       
            // @params:
            //"wf1": randn(in_size, layer_size,    requires_grad=True, dtype=float32),
            //"wf2": randn(layer_size, layer_size, requires_grad=True, dtype=float32),
            //"wk1": randn(in_size, layer_size,    requires_grad=True, dtype=float32),
            //"wk2": randn(layer_size, layer_size, requires_grad=True, dtype=float32),
            //"wi" : randn(in_size, layer_size,    requires_grad=True, dtype=float32),
            
            layer.add(base.randn(i,o));
            layer.add(base.randn(o,o));
            layer.add(base.randn(i,o));
            layer.add(base.randn(o,o));
            layer.add(base.randn(i,o));
            
            gru.add(layer);
            
        }
      
        return gru;
        
    }
    
    ArrayList<ArrayList<Double[][]>> make_lstm(int in_size, ArrayList<Integer> hidden_sizes, int out_size) {
      
        ArrayList<ArrayList<Double[][]>> lstm = new ArrayList();
        
        int hm_layers = hidden_sizes.size()+1;
        
        for (int l = 0; l < hm_layers; l++) {
          
            int i, o;
            ArrayList<Double[][]> layer = new ArrayList();
            
            if      (l == 0)           { i = in_size ; o = hidden_sizes.get(0); }
            else if (l == hm_layers-1) { i = hidden_sizes.get(l-1) ; o = hidden_sizes.get(l); }
            else                       { i = hidden_sizes.get(l) ; o = out_size; }
            
            // @params:
            //"wf1": randn(in_size, layer_size,    requires_grad=True, dtype=float32),
            //"wf2": randn(layer_size, layer_size, requires_grad=True, dtype=float32),
            //"wk1": randn(in_size, layer_size,    requires_grad=True, dtype=float32),
            //"wk2": randn(layer_size, layer_size, requires_grad=True, dtype=float32),
            //"wi1": randn(in_size, layer_size,    requires_grad=True, dtype=float32),
            //"wi2": randn(layer_size, layer_size, requires_grad=True, dtype=float32),
            //"ws1": randn(in_size, layer_size,    requires_grad=True, dtype=float32),
            //"ws2": randn(layer_size, layer_size, requires_grad=True, dtype=float32),
            
            layer.add(base.randn(i,o));
            layer.add(base.randn(o,o));
            layer.add(base.randn(i,o));
            layer.add(base.randn(o,o));
            layer.add(base.randn(i,o));
            layer.add(base.randn(o,o));
            layer.add(base.randn(i,o));
            layer.add(base.randn(o,o));
            
            lstm.add(layer);
            
        }
      
        return lstm;
        
    }
    
    // operations
    
      // respond_gru # todo : make these specific ones into classes
      
      // respond_lstm
      
      // propogate_gru
      
      // propogate_lstm
      
      // zero_states # todo : state is also an attribute of the class.
    
    // helpers
    
      // save
      
      // load
    
}
