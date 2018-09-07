package assignment8;
import java.util.*;

public class ans3 {
public static void main(String args[])
{
	Scanner sc=new Scanner(System.in);
	System.out.println("Enter a number you want to convert");
	int a=sc.nextInt();
	System.out.println("\nAfter conversion to a string\n");
	
	String b=Integer.toString(a);
	System.out.println("Integer.toString(a): "+b+b.getClass().getName());
	
	String c=String.valueOf(a);
	System.out.println("String.valueOf(a): "+c+c.getClass().getName());
	
	Integer e=new Integer(a);
	String f=e.toString();
	System.out.println("intInstance.toString(): "+f+f.getClass().getName());
	
	System.out.println("\n**************************************************************************\n");
	System.out.println("Enter a String you want to convert");
	
	String s1=sc.next();
	Integer i1=new Integer(s1).intValue();
	System.out.println("Integer i1=new Integer(s1).intValue(): "+i1+i1.getClass().getName());
	
	int i2 = Integer.valueOf(s1);
	System.out.println("Integer.valueOf(s1): "+i2);
	
	
}
}
