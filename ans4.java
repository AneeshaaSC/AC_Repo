class a_parent
{
	float a,b;
	final float arithmetic()
	{
		return a/b;
	}
}
class a_child extends a_parent
{
	float arithmetic_2()
	{
		return a%b;
	}
}
public class ans4 {

	public static void main(String args[])
	{
		a_parent p = new a_parent();
		p.a=7;
		p.b=9;
		System.out.println("Division result:"+p.arithmetic());
		a_child c = new a_child();
		c.a=7;
		c.b=9;
		System.out.println("Modulus result:"+c.arithmetic_2());
		
	}
}
// i had to rename the class as a_parent and a_child , else i get an error: type parent is already defined, type child is already defined