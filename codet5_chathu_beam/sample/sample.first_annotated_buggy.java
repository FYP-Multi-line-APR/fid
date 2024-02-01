public class BuggyCode {
    public static void main(String[] args) {
        int x = 5;
        int y = 0;
        
      
        int result = divide(x, y);
       
        
        System.out.println("Result: " + result);
    }
    
    public static int divide(int a, int b) {
        <BUG>
        int result = a / b;
        return result;
        </BUG>
    }
}