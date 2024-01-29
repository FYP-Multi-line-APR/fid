public class FixedCode {
    public static void main(String[] args) {
        int x = 5;
        int y = 0;
        
        <BUG>
        int result = divide(x, y);
        </BUG>
        
        System.out.println("Result: " + result);
    }
    
    public static int divide(int a, int b) {
        // Add a check for zero denominator before performing division
        if (b != 0) {
            int result = a / b;
            return result;
        } else {
            throw new IllegalArgumentException("Denominator cannot be zero.");
        }
    }
}
