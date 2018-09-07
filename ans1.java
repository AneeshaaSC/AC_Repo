package assignment5;
import java.util.*;

class familia
{
	static int family_member_count;
	
	
	static void counter()
	{
		
		family_member_count++;

	}
	static void display_count()
	{
		System.out.println("Number of people in the family= "+family_member_count);
	}
}

public class ans1 {
	
	public static void main (String args[])
	{
		

	//familia f1=new familia();
	Scanner sc=new Scanner(System.in);
	String name=" ";
	while (!name.equals("stop"))
		{
		System.out.println("Enter family member name");
		name=sc.next();
		familia.counter();
		}
	familia.display_count();
	
	}

}
