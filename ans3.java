package assignment7;

class ineligibility_exp extends Exception // why warning or info? what does it mean?
{
	ineligibility_exp()
	{
		System.out.println("Student is not eligible for registration. He/she must be less than 12 years of age and weigh less than 40kgs");
	}
}

class students
{
	int age;
	float weight;
	
	students(int age,float weight)
	{
		this.age=age;
		this.weight=weight;
	}

	void check_eligibility() throws ineligibility_exp
	{
		if (age>12 && weight >40.0) 
		{
			throw new ineligibility_exp(); // why new?
		}
		else
		{
			System.out.println("Student successfully registered.");
		}
	}
}

public class ans3 {
	public static void main(String args[])
	{
		students s1=new students(11,39);
		students s2=new students(15,56);
		try
		{
			s1.check_eligibility();
		}
		catch(ineligibility_exp e)
		{
			
		}
		finally
		{
			System.out.println("Student age:"+s1.age);
			System.out.println("Student age:"+s1.weight);
		}
		try
		{
			s2.check_eligibility();
		}
		catch(ineligibility_exp e) // why the argument or object creation?
		{
			
		}
		finally
		{
			System.out.println("Student age:"+s2.age);
			System.out.println("Student age:"+s2.weight);
		}
	}

}
